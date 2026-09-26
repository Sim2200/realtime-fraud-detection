"""Spark Structured Streaming: score transactions from Kafka with the ONNX model.

Input  topic `transactions`: JSON {event_id, features: [30 floats], produced_at (ISO ms), label}
Output topic `alerts`:       JSON for every event with score >= THRESHOLD, plus the
                             scoring time, and a Parquet sink under /opt/out/scored/
                             with every scored event (score, is_alert, produced_at, scored_at)
                             so end-to-end latency can be computed offline.

The model is a broadcast ONNX file scored in a pandas UDF (one onnxruntime session
per executor Python worker, batches of rows), which is the practical way to serve
a tree model inside Spark without per-row Python overhead.

    spark-submit /opt/jobs/score_stream.py --model /opt/model/best_tree.onnx --threshold 0.5
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="/opt/model/best_tree.onnx")
ap.add_argument("--threshold", type=float, default=0.5)
ap.add_argument("--bootstrap", default="kafka:29092")
ap.add_argument("--out", default="/opt/out")
ap.add_argument("--trigger", default="1 second")
a = ap.parse_args()

spark = (SparkSession.builder.appName("fraud-score-stream")
         .config("spark.sql.shuffle.partitions", "4")
         .config("spark.sql.execution.arrow.maxRecordsPerBatch", "2000")
         .getOrCreate())
spark.sparkContext.setLogLevel("WARN")

model_bytes = spark.sparkContext.broadcast(open(a.model, "rb").read())

_session = None


def _get_session():
    global _session
    if _session is None:
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        _session = ort.InferenceSession(model_bytes.value, opts, providers=["CPUExecutionProvider"])
    return _session


@F.pandas_udf(T.DoubleType())
def score_udf(features: pd.Series) -> pd.Series:
    sess = _get_session()
    x = np.stack(features.to_numpy()).astype(np.float32)
    out = sess.run([sess.get_outputs()[-1].name], {"input": x})[0]
    prob = out[:, 1] if out.ndim == 2 and out.shape[1] == 2 else out.reshape(-1)
    return pd.Series(prob.astype(float))


schema = T.StructType([
    T.StructField("event_id", T.LongType()),
    T.StructField("features", T.ArrayType(T.FloatType())),
    T.StructField("produced_at", T.StringType()),
    T.StructField("label", T.IntegerType()),
])

events = (spark.readStream.format("kafka")
          .option("kafka.bootstrap.servers", a.bootstrap)
          .option("subscribe", "transactions")
          .option("startingOffsets", "earliest")
          .option("maxOffsetsPerTrigger", "20000")
          .load()
          .select(F.from_json(F.col("value").cast("string"), schema).alias("e"))
          .select("e.*"))

scored = (events
          .withColumn("score", score_udf(F.col("features")))
          .withColumn("is_alert", F.col("score") >= F.lit(a.threshold))
          .withColumn("scored_at", F.date_format(F.current_timestamp(), "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'"))
          .drop("features"))

# 1. every scored event to Parquet (for throughput/latency measurement and the drift check)
sink = (scored.writeStream.format("parquet")
        .option("path", f"{a.out}/scored")
        .option("checkpointLocation", f"{a.out}/checkpoint-parquet")
        .trigger(processingTime=a.trigger)
        .outputMode("append")
        .start())

# 2. alerts back to Kafka
alerts = (scored.filter("is_alert")
          .select(F.col("event_id").cast("string").alias("key"),
                  F.to_json(F.struct("event_id", "score", "label", "produced_at", "scored_at")).alias("value"))
          .writeStream.format("kafka")
          .option("kafka.bootstrap.servers", a.bootstrap)
          .option("topic", "alerts")
          .option("checkpointLocation", f"{a.out}/checkpoint-alerts")
          .trigger(processingTime=a.trigger)
          .start())

spark.streams.awaitAnyTermination()
