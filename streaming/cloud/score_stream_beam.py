"""Apache Beam job: score transactions from Pub/Sub with the ONNX model, on Dataflow.

The Spark job's twin (streaming/jobs/score_stream.py). Input messages carry the same JSON
{event_id, features: [29 floats], produced_at, label}; the `event_time` attribute is the
element timestamp. Every event is scored in batches by one onnxruntime session per worker
process (loaded from GCS in DoFn.setup) and written to BigQuery `fraud.scores` with
score, is_alert, produced_at and scored_at, so end-to-end latency is scored_at - produced_at.

A second branch keeps a one-minute event-time window per score and writes
`fraud.score_windows`: events, alerts, mean score, and the PSI of the window's score
distribution against the validation-window reference (same definition as fraud.drift.psi,
inlined because the workers only have this file). That is the drift monitor of section 9
running online.

Submitted by streaming/cloud/run_experiment.py.
"""

import argparse
import json
from datetime import datetime, timezone

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.transforms import trigger, window
from apache_beam.transforms.util import BatchElements

SCORES_SCHEMA = ("event_id:INT64,score:FLOAT64,is_alert:BOOL,label:INT64,produced_at:TIMESTAMP,"
                 "scored_at:TIMESTAMP")
WINDOWS_SCHEMA = ("window_start:TIMESTAMP,window_end:TIMESTAMP,events:INT64,alerts:INT64,mean_score:FLOAT64,"
                  "score_psi:FLOAT64,emitted_at:TIMESTAMP")


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def parse(msg):
    try:
        return json.loads(msg.data.decode())
    except (ValueError, UnicodeDecodeError):
        return None


class Score(beam.DoFn):
    """One onnxruntime session per worker process; scores a batch of events at a time."""

    def __init__(self, model_uri: str, threshold: float):
        self.model_uri, self.threshold = model_uri, threshold

    def setup(self):
        import numpy as np
        import onnxruntime as ort
        from apache_beam.io.gcp.gcsio import GcsIO
        self.np = np
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        with GcsIO().open(self.model_uri, "rb") as f:
            self.session = ort.InferenceSession(f.read(), opts, providers=["CPUExecutionProvider"])
        self.out_name = self.session.get_outputs()[-1].name

    def process(self, events):
        x = self.np.array([e["features"] for e in events], dtype=self.np.float32)
        out = self.session.run([self.out_name], {"input": x})[0]
        prob = out[:, 1] if out.ndim == 2 and out.shape[1] == 2 else out.reshape(-1)
        stamp = now_iso()
        for e, p in zip(events, prob.tolist()):
            yield {"event_id": int(e["event_id"]), "score": p, "is_alert": p >= self.threshold, "label": int(e["label"]),
                   "produced_at": e["produced_at"], "scored_at": stamp}


class WindowStats(beam.CombineFn):
    """Count, alert count, score sum and a score histogram on the reference edges."""

    def __init__(self, edges, reference):
        self.edges, self.reference = edges, reference

    def create_accumulator(self):
        return [0, 0, 0.0, [0] * (len(self.edges) - 1)]

    def _bin(self, s):
        for i in range(len(self.edges) - 1):
            if s < self.edges[i + 1]:
                return i
        return len(self.edges) - 2

    def add_input(self, acc, row):
        acc[0] += 1
        acc[1] += int(row["is_alert"])
        acc[2] += row["score"]
        acc[3][self._bin(row["score"])] += 1
        return acc

    def merge_accumulators(self, accs):
        out = self.create_accumulator()
        for a in accs:
            out[0] += a[0]
            out[1] += a[1]
            out[2] += a[2]
            out[3] = [x + y for x, y in zip(out[3], a[3])]
        return out

    def extract_output(self, acc):
        import math
        n, alerts, total, hist = acc
        eps = 1e-4
        psi = 0.0
        if n:
            for h, r in zip(hist, self.reference):
                p_cur, p_ref = max(h / n, eps), max(r, eps)
                psi += (p_cur - p_ref) * math.log(p_cur / p_ref)
        return {"events": n, "alerts": alerts, "mean_score": (total / n) if n else 0.0, "score_psi": psi}


class ToWindowRow(beam.DoFn):
    def process(self, stats, w=beam.DoFn.WindowParam):
        if not stats["events"]:
            return
        fmt = lambda ts: datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")  # noqa: E731
        yield {"window_start": fmt(w.start), "window_end": fmt(w.end), **{k: (round(v, 6) if isinstance(v, float) else v)
                                                                            for k, v in stats.items()},
               "emitted_at": now_iso()}


def run(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subscription", required=True)
    ap.add_argument("--model_uri", required=True)
    ap.add_argument("--reference_json", required=True, help="JSON string: {edges: [...], reference: [...]}")
    ap.add_argument("--threshold", type=float, required=True)
    ap.add_argument("--scores_table", required=True)
    ap.add_argument("--windows_table", required=True)
    a, beam_args = ap.parse_known_args(argv)
    ref = json.loads(a.reference_json)
    opts = PipelineOptions(beam_args, streaming=True, save_main_session=True)
    p = beam.Pipeline(options=opts)
    scored = (p
              | "read" >> beam.io.ReadFromPubSub(subscription=a.subscription, with_attributes=True,
                                                 timestamp_attribute="event_time")
              | "parse" >> beam.Map(parse)
              | "valid" >> beam.Filter(lambda e: e is not None)
              | "batch" >> BatchElements(min_batch_size=64, max_batch_size=2000, max_batch_duration_secs=0.5)
              | "score" >> beam.ParDo(Score(a.model_uri, a.threshold)))
    (scored
     | "scores->bq" >> beam.io.WriteToBigQuery(a.scores_table, schema=SCORES_SCHEMA,
                                                write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND,
                                                create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED,
                                                method=beam.io.WriteToBigQuery.Method.STREAMING_INSERTS))
    (scored
     | "1min" >> beam.WindowInto(window.FixedWindows(60), trigger=trigger.AfterWatermark(),
                                 accumulation_mode=trigger.AccumulationMode.DISCARDING, allowed_lateness=0)
     | "stats" >> beam.CombineGlobally(WindowStats(ref["edges"], ref["reference"])).without_defaults()
     | "window-row" >> beam.ParDo(ToWindowRow())
     | "windows->bq" >> beam.io.WriteToBigQuery(a.windows_table, schema=WINDOWS_SCHEMA,
                                                 write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND,
                                                 create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED,
                                                 method=beam.io.WriteToBigQuery.Method.STREAMING_INSERTS))
    p.run()


if __name__ == "__main__":
    run()
