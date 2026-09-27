"""Real-time scoring on Google Cloud: Pub/Sub -> Dataflow (ONNX) -> BigQuery, measured end to end.

Mirrors streaming/run_experiment.py (Kafka -> Spark -> Parquet):

1. recreate the two BigQuery tables and drain the subscription,
2. launch the Beam job on Dataflow (onnxruntime installed on the workers from a requirements file),
   and wait until a warm-up event has been scored and is queryable (worker harness up),
3. replay the test window with publish.py at the target rate,
4. poll BigQuery until every event is scored (or no progress for 60 s), sampling the count and the
   newest produced_at each time so "visible lag" can be reported alongside per-event latency,
5. cancel the job, read its resource counters, pull every scored row and compare with local
   onnxruntime scores on the same rows; evaluate alerts against the labels; write
   results/dataflow_scoring.json.

    python streaming/cloud/run_experiment.py --project <id> --rate 2000 --seconds 120
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from google.cloud import bigquery

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fraud import data as D  # noqa: E402

PY = sys.executable
REGION = "us-central1"
PRICE_VCPU_H, PRICE_GB_H, PRICE_SE_GB = 0.069, 0.003557, 0.018


def sh(*args: str, check: bool = True, capture: bool = False) -> str:
    r = subprocess.run(args, text=True, capture_output=capture, check=check)
    return r.stdout if capture else ""


def active_job(project: str, name: str) -> dict | None:
    jobs = json.loads(sh("gcloud", "dataflow", "jobs", "list", "--project", project, "--region", REGION, "--status=active",
                         f"--filter=name={name}", "--format=json", capture=True))
    return jobs[0] if jobs else None


def job_metrics(project: str, job_id: str) -> dict:
    try:
        import google.auth
        import google.auth.transport.requests
        import requests
        creds, _ = google.auth.default()
        creds.refresh(google.auth.transport.requests.Request())
        r = requests.get(f"https://dataflow.googleapis.com/v1b3/projects/{project}/locations/{REGION}/jobs/{job_id}/metrics",
                         headers={"Authorization": f"Bearer {creds.token}"}, timeout=30)
        want = ("TotalVcpuTime", "TotalMemoryUsage", "TotalStreamingDataProcessed", "CurrentVcpuCount")
        return {m["name"]["name"]: m.get("scalar") for m in r.json().get("metrics", [])
                if m["name"]["name"] in want and "tentative" not in (m["name"].get("context") or {})}
    except Exception as e:  # noqa: BLE001
        print("   metrics unavailable:", e)
        return {}


def threshold_raw() -> float:
    """The same raw-score threshold the Spark experiment used (results/streaming_paced.json)."""
    for f in ("results/streaming_paced.json", "results/streaming.json"):
        p = Path(f)
        if p.exists():
            return float(json.loads(p.read_text())["threshold_raw"])
    raise SystemExit("run the local streaming experiment first (it derives the raw threshold)")


def local_scores(x: np.ndarray, model: Path) -> np.ndarray:
    import onnxruntime as ort
    s = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    out = s.run([s.get_outputs()[-1].name], {"input": x.astype(np.float32)})[0]
    return out[:, 1] if out.ndim == 2 and out.shape[1] == 2 else out.reshape(-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--rate", type=float, default=2000)
    ap.add_argument("--seconds", type=float, default=120)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--max-workers", type=int, default=2)
    ap.add_argument("--machine-type", default="e2-standard-2")
    ap.add_argument("--out", default="results/dataflow_scoring.json")
    a = ap.parse_args()
    P = a.project
    bucket_data, bucket_df = f"{P}-fraud", f"{P}-dataflow"
    scores_t, windows_t = f"{P}.fraud.scores", f"{P}.fraud.score_windows"
    thr = threshold_raw()
    client = bigquery.Client(project=P)
    name = f"fraud-scoring-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"

    print("1. tables + subscription")
    client.query(f"""CREATE OR REPLACE TABLE `{scores_t}` (event_id INT64, score FLOAT64, is_alert BOOL, label INT64,
                     produced_at TIMESTAMP, scored_at TIMESTAMP)""").result()
    client.query(f"""CREATE OR REPLACE TABLE `{windows_t}` (window_start TIMESTAMP, window_end TIMESTAMP, events INT64,
                     alerts INT64, mean_score FLOAT64, score_psi FLOAT64, emitted_at TIMESTAMP)""").result()
    sh("gcloud", "pubsub", "subscriptions", "seek", "transactions-beam", "--project", P,
       f"--time={datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}", capture=True)
    reference = json.loads(sh("gcloud", "storage", "cat", f"gs://{bucket_data}/model/score_reference.json", capture=True))

    print("2. launch Dataflow job", name, f"(threshold raw {thr:.5f})")
    req = Path("/tmp/fraud_beam_requirements.txt")
    req.write_text("onnxruntime==1.20.1\nnumpy<2\n")
    t_launch = time.time()
    sh(PY, "streaming/cloud/score_stream_beam.py",
       "--subscription", f"projects/{P}/subscriptions/transactions-beam", "--model_uri", f"gs://{bucket_data}/model/best_tree.onnx",
       "--reference_json", json.dumps({"edges": reference["edges"], "reference": reference["reference"]}),
       "--threshold", str(thr), "--scores_table", scores_t.replace(".", ":", 1), "--windows_table", windows_t.replace(".", ":", 1),
       "--runner", "DataflowRunner", "--project", P, "--region", REGION, "--job_name", name,
       "--temp_location", f"gs://{bucket_df}/tmp", "--staging_location", f"gs://{bucket_df}/staging",
       "--requirements_file", str(req), "--machine_type", a.machine_type, "--num_workers", str(a.workers),
       "--max_num_workers", str(a.max_workers), "--enable_streaming_engine", "--experiments", "use_runner_v2")
    while True:
        j = active_job(P, name)
        st = (j.get("state") or j.get("currentState") or "?").replace("JOB_STATE_", "").title() if j else "?"
        print("  ", st, flush=True)
        if st == "Running":
            break
        if st in ("Failed", "Cancelled", "Done"):
            raise SystemExit("job did not start")
        time.sleep(15)
    t_running = time.time()

    print("   warm-up: 2 events/s until one is scored and queryable")
    warm = subprocess.Popen([PY, "streaming/cloud/publish.py", "--project", P, "--rate", "2", "--seconds", "900"],
                            stdout=subprocess.DEVNULL)
    while not list(client.query(f"SELECT 1 FROM `{scores_t}` LIMIT 1").result()):
        time.sleep(5)
    warm.terminate()
    t_ready = time.time()
    print(f"   first scored row {t_ready - t_running:.0f} s after RUNNING")
    time.sleep(5)
    client.query(f"TRUNCATE TABLE `{scores_t}`").result()
    client.query(f"TRUNCATE TABLE `{windows_t}`").result()

    print(f"3. publish {a.rate:g} events/s for {a.seconds:g} s + 4. poll BigQuery")
    t0 = time.time()
    pub = threading.Thread(target=lambda: sh(PY, "streaming/cloud/publish.py", "--project", P, "--rate", str(a.rate),
                                             "--seconds", str(a.seconds)))
    pub.start()
    samples, last_n, last_change = [], -1, time.time()
    expected = None
    while True:
        row = list(client.query(f"SELECT count(*) n, max(produced_at) newest FROM `{scores_t}`").result())[0]
        now = datetime.now(timezone.utc)
        lag = (now - row.newest).total_seconds() if row.newest else None
        samples.append({"t": round(time.time() - t0, 1), "scored": int(row.n), "visible_lag_s": round(lag, 1) if lag else None})
        print(f"   t={samples[-1]['t']:6.1f}s scored {row.n:,} visible lag {samples[-1]['visible_lag_s']} s", flush=True)
        if not pub.is_alive() and expected is None:
            expected = json.loads(Path("results/pubsub_publish.json").read_text())["events_sent"]
        if row.n != last_n:
            last_n, last_change = row.n, time.time()
        if expected is not None and (row.n >= expected or time.time() - last_change > 60):
            break
        time.sleep(5)
    pub.join()
    drain_seconds = time.time() - t0
    publisher = json.loads(Path("results/pubsub_publish.json").read_text())

    print("5. cancel, metrics, compare")
    j = active_job(P, name)
    totals = job_metrics(P, j["id"]) if j else {}
    if j:
        sh("gcloud", "dataflow", "jobs", "cancel", j["id"], "--project", P, "--region", REGION, check=False, capture=True)
    rows = list(client.query(f"SELECT event_id, score, is_alert, label, produced_at, scored_at FROM `{scores_t}`").result())
    ids = np.array([r.event_id for r in rows])
    remote = np.array([r.score for r in rows])
    lat = np.array([(r.scored_at - r.produced_at).total_seconds() * 1000 for r in rows])
    split = D.time_split(D.load())
    local = local_scores(split.x_test[ids], Path("results/artifacts/best_tree.onnx"))
    y = split.y_test[ids]
    alert = np.array([r.is_alert for r in rows])
    tp, fp, fn = int((alert & (y == 1)).sum()), int((alert & (y == 0)).sum()), int((~alert & (y == 1)).sum())
    windows = [dict(r) for r in client.query(f"SELECT * FROM `{windows_t}` ORDER BY window_start").result()]
    vcpu_h = float(totals.get("TotalVcpuTime") or 0) / 3600
    gb_h = float(totals.get("TotalMemoryUsage") or 0) / 1024 / 3600
    se_gb = float(totals.get("TotalStreamingDataProcessed") or 0) / 1024
    res = {
        "job": {"name": name, "region": REGION, "machine_type": a.machine_type, "workers": a.workers,
                "seconds_to_running": round(t_running - t_launch), "seconds_to_first_scored_row": round(t_ready - t_running),
                "wall_seconds": round(time.time() - t_launch)},
        "publisher": publisher,
        "events_scored": int(len(rows)), "duplicates": int(len(rows) - len(set(ids.tolist()))),
        "wall_seconds_publish_to_drained": round(drain_seconds, 1),
        "dataflow_throughput_events_per_second": round(len(rows) / drain_seconds, 1),
        "latency_ms_scored_minus_produced": {"p50": round(float(np.percentile(lat, 50)), 1), "p95": round(float(np.percentile(lat, 95)), 1),
                                             "p99": round(float(np.percentile(lat, 99)), 1), "max": round(float(lat.max()), 1)},
        "visible_lag_samples": samples,
        "threshold_raw": thr,
        "agreement_with_local_onnx": {"max_abs_score_diff": float(np.abs(remote - local).max()),
                                      "alerts_identical": bool(np.array_equal(alert, local >= thr))},
        "alerts": {"count": int(alert.sum()), "tp": tp, "fp": fp, "fn": fn,
                   "precision": round(tp / max(1, tp + fp), 4), "recall": round(tp / max(1, tp + fn), 4)},
        "score_windows": [{k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in w.items()} for w in windows],
        "dataflow_metrics": totals,
        "cost_estimate_usd": {"vcpu_hours": round(vcpu_h, 3), "memory_gb_hours": round(gb_h, 3), "streaming_data_gb": round(se_gb, 3),
                              "total": round(vcpu_h * PRICE_VCPU_H + gb_h * PRICE_GB_H + se_gb * PRICE_SE_GB, 4),
                              "note": "Dataflow list prices, us-central1; Pub/Sub and BigQuery streaming inserts are cents at this volume"},
    }
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k not in ("visible_lag_samples", "score_windows", "publisher")}, indent=2))


if __name__ == "__main__":
    main()
