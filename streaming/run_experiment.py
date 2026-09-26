"""The streaming experiment end to end (requires `make stream-up` and `make onnx`):

1. reset topics and the Parquet sink,
2. submit the Spark scoring job (background, inside the spark container),
3. replay the test window with the producer at the target rate,
4. wait until Spark has scored every event, stop the job,
5. read the Parquet sink: throughput, per-event latency p50/p95/p99, alert precision/recall
   against the labels at the threshold; write results/streaming.json.

    python streaming/run_experiment.py --rate 2000 --seconds 120
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

COMPOSE = ["docker", "compose", "-f", "streaming/docker-compose.yml"]
PY = sys.executable
OUT = Path("streaming/out")


def sh(*args: str, check: bool = True, capture: bool = False, background: bool = False):
    if background:
        return subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    r = subprocess.run(args, check=check, text=True, capture_output=capture)
    return r.stdout if capture else ""


def threshold_from_results() -> float:
    ev = Path("results/evaluation.json")
    if ev.exists():
        return float(json.loads(ev.read_text())["cost"]["threshold_from_validation"])
    return 0.5


def parse_ts(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, format="%Y-%m-%dT%H:%M:%S.%fZ", utc=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", type=float, default=2000)
    ap.add_argument("--seconds", type=float, default=120)
    ap.add_argument("--out", default="results/streaming.json")
    a = ap.parse_args()
    thr = threshold_from_results()
    # The Spark UDF scores the raw model; apply the calibrator's monotone map to the
    # threshold instead (isotonic is monotone, so ranking is unchanged).
    raw_thr = raw_threshold(thr)

    print("== reset topics and sink")
    topics = [*COMPOSE, "exec", "-T", "kafka", "/opt/kafka/bin/kafka-topics.sh", "--bootstrap-server", "kafka:29092"]
    for t in ("transactions", "alerts"):
        sh(*topics, "--delete", "--if-exists", "--topic", t, capture=True)
    time.sleep(3)
    for t in ("transactions", "alerts"):
        sh(*topics, "--create", "--if-not-exists", "--topic", t, "--partitions", "4", "--replication-factor", "1", capture=True)
    # Clear the sink's contents but keep the directory: it is bind-mounted into the
    # Spark container, and deleting it would leave the container writing to a dead inode.
    OUT.mkdir(parents=True, exist_ok=True)
    for child in OUT.iterdir():
        shutil.rmtree(child) if child.is_dir() else child.unlink()

    print(f"== submit Spark job (threshold raw={raw_thr:.4f})")
    job = sh(*COMPOSE, "exec", "-T", "spark", "/opt/spark/bin/spark-submit", "--master", "local[4]",
             "--driver-memory", "2g", "/opt/jobs/score_stream.py", "--threshold", str(raw_thr), background=True)
    time.sleep(25)  # Spark startup + Kafka source discovery

    print(f"== producer: {a.rate:g} events/s for {a.seconds:g}s")
    t0 = time.time()
    sh(PY, "streaming/producer.py", "--rate", str(a.rate), "--seconds", str(a.seconds))
    producer = json.loads(Path("results/producer.json").read_text())

    print("== waiting for Spark to drain")
    expected = producer["events_sent"]
    last, stable = -1, 0
    for _ in range(120):
        time.sleep(3)
        n = count_scored()
        print(f"   scored {n:,}/{expected:,}", flush=True)
        if n >= expected:
            break
        stable = stable + 1 if n == last else 0
        last = n
        if stable >= 10:
            print("   no progress for 30 s, stopping")
            break
    drain_seconds = time.time() - t0
    job.terminate()
    sh(*COMPOSE, "exec", "-T", "spark", "pkill", "-f", "score_stream.py", check=False, capture=True)

    df = pd.read_parquet(OUT / "scored")
    lat = (parse_ts(df["scored_at"]) - parse_ts(df["produced_at"])).dt.total_seconds() * 1000
    y, alert = df["label"].to_numpy(), df["is_alert"].to_numpy()
    tp, fp, fn = int((alert & (y == 1)).sum()), int((alert & (y == 0)).sum()), int((~alert & (y == 1)).sum())
    res = {
        "producer": producer,
        "events_scored": int(len(df)),
        "wall_seconds_produce_to_drained": round(drain_seconds, 1),
        "spark_throughput_events_per_second": round(len(df) / drain_seconds, 1),
        "latency_ms": {"p50": round(float(np.percentile(lat, 50)), 1), "p95": round(float(np.percentile(lat, 95)), 1),
                       "p99": round(float(np.percentile(lat, 99)), 1), "max": round(float(lat.max()), 1)},
        "threshold_raw": raw_thr, "threshold_calibrated": thr,
        "alerts": {"count": int(alert.sum()), "tp": tp, "fp": fp, "fn": fn,
                   "precision": round(tp / max(1, tp + fp), 4), "recall": round(tp / max(1, tp + fn), 4)},
        "spark": "local[4], 1 s micro-batches, pandas UDF over a broadcast ONNX model",
    }
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "producer"}, indent=2))


def count_scored() -> int:
    files = list((OUT / "scored").glob("*.parquet"))
    if not files:
        return 0
    try:
        import pyarrow.parquet as pq
        return sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    except Exception:  # noqa: BLE001
        return 0


def raw_threshold(calibrated: float) -> float:
    """Invert the isotonic calibrator: smallest raw score whose calibrated value >= threshold."""
    cal = Path("results/artifacts/calibrator.joblib")
    if not cal.exists():
        return calibrated
    import joblib
    iso = joblib.load(cal)
    grid = np.linspace(0, 1, 100_001)
    mapped = iso.predict(grid)
    idx = np.argmax(mapped >= calibrated)
    return float(grid[idx]) if mapped[idx] >= calibrated else 1.0


if __name__ == "__main__":
    main()
