"""Replay the test window into Pub/Sub topic, in time order.

Each event: {event_id, features: [29 floats in fraud.data.FEATURES order], produced_at, label}.
`produced_at` is the wall-clock send time; the Beam job stamps `scored_at`, so
end-to-end latency = scored_at - produced_at per event. Labels ride along only so
the alerts can be evaluated afterwards; the model never sees them.

    python streaming/cloud/publish.py --project my-project --rate 2000 --seconds 120      # paced
    python streaming/cloud/publish.py --project my-project --rate 0                       # as fast as possible
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import wait
from datetime import datetime, timezone
from pathlib import Path

from google.cloud import pubsub_v1

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fraud import data as D  # noqa: E402


def iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.") + f"{int(ts * 1000) % 1000:03d}Z"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True, help="GCP project ID")
    ap.add_argument("--topic", default="transactions", help="Pub/Sub topic name")
    ap.add_argument("--rate", type=float, default=2000, help="events/second; 0 = unthrottled")
    ap.add_argument("--seconds", type=float, default=120)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    split = D.time_split(D.load())
    x, y = split.x_test, split.y_test
    n = len(x) if not a.limit else min(a.limit, len(x))

    publisher = pubsub_v1.PublisherClient(batch_settings=pubsub_v1.types.BatchSettings(
        max_messages=500,
        max_bytes=1_000_000,
        max_latency=0.05,
    ))

    topic_path = publisher.topic_path(a.project, a.topic)

    start = time.time()
    sent = 0
    futures = []

    for i in range(n):
        now = time.time()
        if now - start > a.seconds:
            break
        if a.rate > 0:
            due = start + sent / a.rate
            if due > now:
                time.sleep(due - now)
                now = time.time()

        produced_at = iso(now)
        event = {
            "event_id": i,
            "features": [round(float(v), 6) for v in x[i]],
            "produced_at": produced_at,
            "label": int(y[i])
        }

        future = publisher.publish(
            topic_path,
            json.dumps(event).encode(),
            event_time=produced_at,
        )
        futures.append(future)

        sent += 1
        if sent % 20000 == 0:
            print(f"  {sent:,} events, {sent / (time.time() - start):,.0f}/s", flush=True)
            wait(futures, return_when='ALL_COMPLETED', timeout=60)
            futures = []

    # Wait on all remaining futures
    if futures:
        wait(futures, return_when='ALL_COMPLETED', timeout=60)

    elapsed = time.time() - start
    res = {"events_sent": sent, "seconds": round(elapsed, 2), "events_per_second": round(sent / elapsed, 1),
           "target_rate": a.rate, "frauds_sent": int(y[:sent].sum())}
    Path("results").mkdir(exist_ok=True)
    Path("results/pubsub_publish.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
