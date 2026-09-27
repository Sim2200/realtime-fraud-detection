"""The fraud model's lifecycle as a Vertex AI Pipeline (Kubeflow Pipelines v2).

    ingest -> validate -> train -> evaluate --(gate)--> register -> deploy -> smoke_test
                 \\-> drift                               \\-> batch_predict

Every component runs in the project's own image (vertex/Dockerfile), so the
pipeline reuses fraud.data / fraud.models / fraud.metrics / fraud.drift rather
than re-implementing them. The gate promotes a model only if its test PR-AUC
clears a floor and is not worse than the version currently in the Model
Registry. Registration, deployment and batch prediction use the same image as
a custom serving container (vertex/serve.py).

    python vertex/pipeline.py compile  --out pipeline.json
    python vertex/pipeline.py run      --project <id>      # submit, wait, write results/vertex_run.json
    python vertex/pipeline.py schedule --project <id>      # weekly retrain schedule
    python vertex/pipeline.py teardown --project <id>      # undeploy, delete endpoint + schedule
"""


import argparse
import json
import os
import time
from pathlib import Path
from typing import NamedTuple

from kfp import compiler, dsl
from kfp.dsl import ClassificationMetrics, Dataset, Input, Metrics, Model, Output

PROJECT = os.environ.get("GCP_PROJECT", "project-1adf2361-a5bc-4d4a-a7f")
REGION = "us-central1"
IMAGE = os.environ.get("PIPELINE_IMAGE", f"{REGION}-docker.pkg.dev/{PROJECT}/fraud/pipeline:v1")
MODEL_NAME = "fraud-detector"
ENDPOINT_NAME = "fraud-endpoint"
SCHEDULE_NAME = "fraud-weekly-retrain"
PR_AUC_FLOOR = 0.75


# ---------------------------------------------------------------- components

@dsl.component(base_image=IMAGE)
def ingest(data_uri: str, train: Output[Dataset], val: Output[Dataset], test: Output[Dataset],
           summary: Output[Metrics]) -> None:
    """Time-ordered 70/10/20 split of the raw table, no shuffling (fraud.data.time_split)."""
    import pandas as pd
    from fraud import data as D
    df = pd.read_parquet(data_uri)
    df[D.TARGET] = df[D.TARGET].astype(int)
    if D.TIME not in df:
        df[D.TIME] = range(len(df))
    s = D.time_split(df)
    for part, out in ((s.train, train), (s.val, val), (s.test, test)):
        part.to_parquet(out.path)
    for name, part in s.summary().items():
        summary.log_metric(f"{name}_rows", part["rows"])
        summary.log_metric(f"{name}_frauds", part["frauds"])
        summary.log_metric(f"{name}_fraud_rate", part["fraud_rate"])


@dsl.component(base_image=IMAGE)
def validate(train: Input[Dataset], report: Output[Metrics]) -> None:
    """Schema and sanity checks on the training slice; fails the run if the data is wrong."""
    import pandas as pd
    from fraud import data as D
    df = pd.read_parquet(train.path)
    problems = []
    missing = [c for c in D.FEATURES + [D.TARGET] if c not in df.columns]
    if missing:
        problems.append(f"missing columns {missing}")
    nulls = int(df[D.FEATURES].isna().sum().sum())
    if nulls:
        problems.append(f"{nulls} null feature values")
    if not set(df[D.TARGET].unique()) <= {0, 1}:
        problems.append("target is not binary")
    rate = float(df[D.TARGET].mean())
    if not 0.0005 <= rate <= 0.05:
        problems.append(f"fraud rate {rate:.5f} outside [0.05%, 5%]")
    if (df["Amount"] < 0).any():
        problems.append("negative amounts")
    report.log_metric("rows", int(len(df)))
    report.log_metric("null_features", nulls)
    report.log_metric("fraud_rate", round(rate, 6))
    report.log_metric("problems", len(problems))
    if problems:
        raise RuntimeError("validation failed: " + "; ".join(problems))


@dsl.component(base_image=IMAGE)
def train(train: Input[Dataset], val: Input[Dataset], model: Output[Model], metrics: Output[Metrics],
          n_jobs: int = 4) -> None:
    """Fit the tree candidates, pick by validation PR-AUC only, save model + meta (features, threshold)."""
    import json
    import os
    import time

    import joblib
    import numpy as np
    import pandas as pd
    from fraud import data as D
    from fraud import metrics as M
    from fraud import models as Mo
    tr, va = pd.read_parquet(train.path), pd.read_parquet(val.path)
    x_tr, y_tr = tr[D.FEATURES].to_numpy(np.float32), tr[D.TARGET].to_numpy(np.int64)
    x_va, y_va = va[D.FEATURES].to_numpy(np.float32), va[D.TARGET].to_numpy(np.int64)
    best, best_t = None, None
    t0 = time.perf_counter()
    for name, strategy in (("xgboost", "weights"), ("lightgbm", "weights"), ("xgboost", "none")):
        t = Mo.train(name, strategy, x_tr, y_tr, x_va, y_va, seed=0, n_jobs=n_jobs)
        v = M.pr_auc(y_va, t.score(x_va))
        metrics.log_metric(f"val_pr_auc_{name}_{strategy}", round(v, 4))
        metrics.log_metric(f"fit_seconds_{name}_{strategy}", t.fit_seconds)
        if best is None or v > best:
            best, best_t = v, t
    thr = M.best_f1_threshold(y_va, best_t.score(x_va))
    os.makedirs(model.path, exist_ok=True)
    joblib.dump(best_t.model, os.path.join(model.path, "model.joblib"))
    meta = {"model": best_t.name, "strategy": best_t.strategy, "features": D.FEATURES, "threshold": float(thr),
            "val_pr_auc": round(float(best), 4), "n_train_rows": best_t.n_train, **best_t.extra}
    with open(os.path.join(model.path, "model_meta.json"), "w") as f:
        json.dump(meta, f)
    model.metadata.update({k: v for k, v in meta.items() if k != "features"})
    metrics.log_metric("selected", f"{best_t.name}/{best_t.strategy}")
    metrics.log_metric("val_pr_auc", round(float(best), 4))
    metrics.log_metric("threshold", round(float(thr), 4))
    metrics.log_metric("train_wall_seconds", round(time.perf_counter() - t0, 1))


@dsl.component(base_image=IMAGE)
def evaluate(test: Input[Dataset], model: Input[Model], project: str, region: str, model_name: str,
             pr_auc_floor: float, metrics: Output[Metrics], curves: Output[ClassificationMetrics],
             instances: Output[Dataset]
             ) -> NamedTuple("Gate", [("passed", bool), ("champion_pr_auc", float), ("test_pr_auc", float)]):
    """Test-set report (fraud.metrics.full_report), ROC curve + confusion matrix for the UI, promotion gate."""
    import json
    import os
    from typing import NamedTuple

    import joblib
    import numpy as np
    import pandas as pd
    from google.cloud import aiplatform
    from sklearn.metrics import confusion_matrix, roc_curve
    from fraud import data as D
    from fraud import metrics as M
    te = pd.read_parquet(test.path)
    x, y = te[D.FEATURES].to_numpy(np.float32), te[D.TARGET].to_numpy(np.int64)
    clf = joblib.load(os.path.join(model.path, "model.joblib"))
    meta = json.load(open(os.path.join(model.path, "model_meta.json")))
    score = clf.predict_proba(x)[:, 1]
    rep = M.full_report(y, score, threshold=meta["threshold"])
    for k, v in rep.items():
        metrics.log_metric(k, v)
    fpr, tpr, thr = roc_curve(y, score)
    keep = np.linspace(0, len(fpr) - 1, min(len(fpr), 200)).astype(int)
    curves.log_roc_curve(fpr[keep].tolist(), tpr[keep].tolist(), np.nan_to_num(thr[keep], posinf=1.0).tolist())
    cm = confusion_matrix(y, score >= meta["threshold"]).tolist()
    curves.log_confusion_matrix(["legit", "fraud"], cm)
    # Batch-prediction input: one JSON array of 29 features per line.
    with open(instances.path, "w") as f:
        for row in x.tolist():
            f.write(json.dumps(row) + "\n")
    # Champion: the newest registered version's recorded test PR-AUC (label, since labels are strings).
    aiplatform.init(project=project, location=region)
    champ = 0.0
    found = aiplatform.Model.list(filter=f'display_name="{model_name}"', order_by="create_time desc")
    if found:
        champ = float((found[0].labels or {}).get("test_pr_auc", "0").replace("_", "."))
    passed = bool(rep["pr_auc"] >= pr_auc_floor and rep["pr_auc"] >= champ - 0.005)
    metrics.log_metric("champion_pr_auc", champ)
    metrics.log_metric("gate_passed", int(passed))
    return NamedTuple("Gate", [("passed", bool), ("champion_pr_auc", float), ("test_pr_auc", float)])(
        passed, champ, float(rep["pr_auc"]))


@dsl.component(base_image=IMAGE)
def register(model: Input[Model], project: str, region: str, model_name: str, image: str, test_pr_auc: float
             ) -> NamedTuple("Registered", [("resource_name", str), ("version_id", str)]):
    """Upload to the Model Registry as a new version of one lineage, served by the custom container."""
    from typing import NamedTuple

    from google.cloud import aiplatform
    aiplatform.init(project=project, location=region)
    parents = aiplatform.Model.list(filter=f'display_name="{model_name}"', order_by="create_time desc")
    m = aiplatform.Model.upload(
        display_name=model_name, artifact_uri=model.uri, parent_model=parents[0].resource_name if parents else None,
        serving_container_image_uri=image, serving_container_predict_route="/predict",
        serving_container_health_route="/health", serving_container_ports=[8080],
        labels={"test_pr_auc": f"{test_pr_auc:.4f}".replace(".", "_"),
                "algorithm": str(model.metadata.get("model", "tree"))},
        version_description=f"test PR-AUC {test_pr_auc:.4f}", is_default_version=True, sync=True)
    return NamedTuple("Registered", [("resource_name", str), ("version_id", str)])(m.resource_name, m.version_id)


@dsl.component(base_image=IMAGE)
def deploy(model_resource: str, project: str, region: str, endpoint_name: str, machine_type: str,
           timing: Output[Metrics]) -> str:
    """Get-or-create the endpoint, deploy the new version with all traffic, undeploy older versions."""
    import time

    from google.cloud import aiplatform
    aiplatform.init(project=project, location=region)
    eps = aiplatform.Endpoint.list(filter=f'display_name="{endpoint_name}"')
    ep = eps[0] if eps else aiplatform.Endpoint.create(display_name=endpoint_name)
    m = aiplatform.Model(model_resource)
    old = [d.id for d in ep.list_models()]
    t0 = time.perf_counter()
    m.deploy(endpoint=ep, machine_type=machine_type, min_replica_count=1, max_replica_count=1, traffic_percentage=100,
             deployed_model_display_name=f"{endpoint_name}-v{m.version_id}", sync=True)
    timing.log_metric("deploy_seconds", round(time.perf_counter() - t0, 1))
    for d in old:
        ep.undeploy(deployed_model_id=d, sync=True)
    timing.log_metric("previous_versions_undeployed", len(old))
    return ep.resource_name


@dsl.component(base_image=IMAGE)
def smoke_test(endpoint_resource: str, test: Input[Dataset], model: Input[Model], project: str, region: str,
               n_requests: int, metrics: Output[Metrics]) -> None:
    """Online predictions: latency distribution and agreement with the local model on the same rows."""
    import json
    import os
    import time

    import joblib
    import numpy as np
    import pandas as pd
    from google.cloud import aiplatform
    from fraud import data as D
    aiplatform.init(project=project, location=region)
    ep = aiplatform.Endpoint(endpoint_resource)
    te = pd.read_parquet(test.path).head(n_requests)
    x = te[D.FEATURES].to_numpy(np.float32)
    clf = joblib.load(os.path.join(model.path, "model.joblib"))
    local = clf.predict_proba(x)[:, 1]
    lat, remote = [], []
    for row in x.tolist():
        t0 = time.perf_counter()
        r = ep.predict(instances=[row])
        lat.append((time.perf_counter() - t0) * 1000)
        remote.append(float(r.predictions[0]["score"]))
    lat = np.array(lat)
    metrics.log_metric("requests", int(len(lat)))
    metrics.log_metric("latency_p50_ms", round(float(np.percentile(lat, 50)), 1))
    metrics.log_metric("latency_p95_ms", round(float(np.percentile(lat, 95)), 1))
    metrics.log_metric("latency_max_ms", round(float(lat.max()), 1))
    metrics.log_metric("max_abs_score_diff_vs_local", round(float(np.abs(np.array(remote) - local).max()), 6))
    t0 = time.perf_counter()
    r = ep.predict(instances=x[:100].tolist())
    metrics.log_metric("batch100_ms", round((time.perf_counter() - t0) * 1000, 1))
    metrics.log_metric("batch100_predictions", len(r.predictions))


@dsl.component(base_image=IMAGE)
def batch_predict(model_resource: str, instances: Input[Dataset], project: str, region: str, bucket: str,
                  machine_type: str, metrics: Output[Metrics]) -> None:
    """A Vertex batch prediction job over the whole test slice, timed end to end."""
    import time

    from google.cloud import aiplatform
    aiplatform.init(project=project, location=region)
    t0 = time.perf_counter()
    job = aiplatform.BatchPredictionJob.create(
        job_display_name="fraud-batch-test", model_name=model_resource, instances_format="jsonl",
        gcs_source=instances.uri, gcs_destination_prefix=f"gs://{bucket}/batch-predictions",
        predictions_format="jsonl", machine_type=machine_type, starting_replica_count=1, max_replica_count=1,
        sync=True)
    wall = time.perf_counter() - t0
    n = sum(1 for _ in open(instances.path))
    metrics.log_metric("rows", n)
    metrics.log_metric("wall_seconds", round(wall, 1))
    metrics.log_metric("rows_per_second_incl_startup", round(n / wall, 1))
    metrics.log_metric("state", str(job.state).split(".")[-1])
    metrics.log_metric("output_dir", job.output_info.gcs_output_directory if job.output_info else "")


@dsl.component(base_image=IMAGE)
def drift(train: Input[Dataset], test: Input[Dataset], metrics: Output[Metrics]) -> None:
    """PSI of every feature, newest slice vs training slice (fraud.drift.feature_drift)."""
    import numpy as np
    import pandas as pd
    from fraud import data as D
    from fraud import drift as Dr
    tr, te = pd.read_parquet(train.path), pd.read_parquet(test.path)
    rows = Dr.feature_drift(tr[D.FEATURES].to_numpy(np.float32), te[D.FEATURES].to_numpy(np.float32), D.FEATURES)
    metrics.log_metric("features", len(rows))
    metrics.log_metric("max_psi", rows[0]["psi"])
    metrics.log_metric("max_psi_feature", rows[0]["feature"])
    metrics.log_metric("moderate_or_worse", sum(r["status"] != "stable" for r in rows))
    for r in rows[:5]:
        metrics.log_metric(f"psi_{r['feature']}", r["psi"])


# ---------------------------------------------------------------- pipeline

@dsl.pipeline(name="fraud-detector-lifecycle")
def pipeline(data_uri: str, project: str = PROJECT, region: str = REGION, bucket: str = f"{PROJECT}-fraud",
             image: str = IMAGE, model_name: str = MODEL_NAME, endpoint_name: str = ENDPOINT_NAME,
             pr_auc_floor: float = PR_AUC_FLOOR, machine_type: str = "n1-standard-2", smoke_requests: int = 200,
             do_deploy: bool = True, do_batch: bool = True):
    ing = ingest(data_uri=data_uri)
    val = validate(train=ing.outputs["train"])
    tr = train(train=ing.outputs["train"], val=ing.outputs["val"]).after(val)
    tr.set_cpu_limit("4").set_memory_limit("8G")
    ev = evaluate(test=ing.outputs["test"], model=tr.outputs["model"], project=project, region=region,
                  model_name=model_name, pr_auc_floor=pr_auc_floor)
    drift(train=ing.outputs["train"], test=ing.outputs["test"])
    with dsl.If(ev.outputs["passed"] == True, name="gate-passed"):  # noqa: E712
        reg = register(model=tr.outputs["model"], project=project, region=region, model_name=model_name, image=image,
                       test_pr_auc=ev.outputs["test_pr_auc"])
        with dsl.If(do_deploy == True, name="deploy"):  # noqa: E712
            dep = deploy(model_resource=reg.outputs["resource_name"], project=project, region=region,
                         endpoint_name=endpoint_name, machine_type=machine_type)
            smoke_test(endpoint_resource=dep.outputs["Output"], test=ing.outputs["test"], model=tr.outputs["model"],
                       project=project, region=region, n_requests=smoke_requests)
        with dsl.If(do_batch == True, name="batch"):  # noqa: E712
            batch_predict(model_resource=reg.outputs["resource_name"], instances=ev.outputs["instances"],
                          project=project, region=region, bucket=bucket, machine_type=machine_type)


# ---------------------------------------------------------------- CLI

def compile_pipeline(out: str) -> None:
    compiler.Compiler().compile(pipeline_func=pipeline, package_path=out)
    print("wrote", out)


def run(project: str, wait: bool = True, **params) -> None:
    from google.cloud import aiplatform
    aiplatform.init(project=project, location=REGION, staging_bucket=f"gs://{project}-fraud")
    pkg = "/tmp/fraud_pipeline.json"
    compile_pipeline(pkg)
    job = aiplatform.PipelineJob(
        display_name="fraud-detector-lifecycle", template_path=pkg, pipeline_root=f"gs://{project}-fraud/pipeline-root",
        parameter_values={"data_uri": f"gs://{project}-fraud/data/creditcard.parquet", "project": project, **params},
        enable_caching=False)
    t0 = time.time()
    job.submit()
    print("submitted", job.resource_name, "\n", job._dashboard_uri())
    if not wait:
        return
    job.wait()
    tasks = []
    for t in job.task_details:
        row = {"task": t.task_name, "state": str(t.state).split(".")[-1],
               "seconds": round((t.end_time - t.start_time).total_seconds(), 1) if t.end_time and t.start_time else None,
               "metrics": {}}
        for name, arts in (t.outputs or {}).items():
            for a in arts.artifacts:
                if a.schema_title == "system.Metrics":
                    row["metrics"].update({k: (v if not hasattr(v, "keys") else dict(v)) for k, v in a.metadata.items()})
        tasks.append(row)
    out = {"job": job.resource_name, "state": str(job.state).split(".")[-1], "wall_seconds": round(time.time() - t0),
           "dashboard": job._dashboard_uri(), "tasks": tasks}
    Path("results").mkdir(exist_ok=True)
    Path("results/vertex_run.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps({t["task"]: (t["state"], t["seconds"]) for t in tasks}, indent=2))


def schedule(project: str) -> None:
    from google.cloud import aiplatform
    aiplatform.init(project=project, location=REGION)
    pkg = "/tmp/fraud_pipeline.json"
    compile_pipeline(pkg)
    job = aiplatform.PipelineJob(display_name="fraud-detector-lifecycle", template_path=pkg,
                                 pipeline_root=f"gs://{project}-fraud/pipeline-root",
                                 parameter_values={"data_uri": f"gs://{project}-fraud/data/creditcard.parquet",
                                                   "project": project, "do_deploy": True, "do_batch": False},
                                 enable_caching=False)
    s = job.create_schedule(display_name=SCHEDULE_NAME, cron="TZ=America/New_York 0 6 * * 1",
                            max_concurrent_run_count=1)
    print("schedule", s.resource_name, "next run", s.next_run_time)


def teardown(project: str) -> None:
    from google.cloud import aiplatform
    aiplatform.init(project=project, location=REGION)
    for s in aiplatform.PipelineJobSchedule.list(filter=f'display_name="{SCHEDULE_NAME}"'):
        s.delete()
        print("deleted schedule", s.resource_name)
    for ep in aiplatform.Endpoint.list(filter=f'display_name="{ENDPOINT_NAME}"'):
        ep.undeploy_all(sync=True)
        ep.delete()
        print("deleted endpoint", ep.resource_name)
    print("models kept in the registry:", [m.resource_name for m in aiplatform.Model.list(filter=f'display_name="{MODEL_NAME}"')])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["compile", "run", "schedule", "teardown"])
    ap.add_argument("--project", default=PROJECT)
    ap.add_argument("--out", default="pipeline.json")
    ap.add_argument("--no-wait", action="store_true")
    ap.add_argument("--no-deploy", action="store_true")
    ap.add_argument("--no-batch", action="store_true")
    a = ap.parse_args()
    if a.command == "compile":
        compile_pipeline(a.out)
    elif a.command == "run":
        run(a.project, wait=not a.no_wait, do_deploy=not a.no_deploy, do_batch=not a.no_batch)
    elif a.command == "schedule":
        schedule(a.project)
    else:
        teardown(a.project)


if __name__ == "__main__":
    main()
