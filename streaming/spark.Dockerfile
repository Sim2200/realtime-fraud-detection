# Spark 3.5 with Python, plus onnxruntime for the scoring UDF and the Kafka
# connector jars pre-fetched so `spark-submit` does not download at run time.
FROM apache/spark:3.5.9-python3

USER root
RUN pip3 install --no-cache-dir onnxruntime==1.19.2 "numpy<2" pandas pyarrow \
 && mkdir -p /opt/out && chown -R spark:spark /opt/out
# Kafka source/sink for Structured Streaming and its transitive dependencies.
ARG SPARK_VER=3.5.9
ARG KAFKA_CLIENT=3.4.1
RUN cd /opt/spark/jars \
 && curl -fsSLO https://repo1.maven.org/maven2/org/apache/spark/spark-sql-kafka-0-10_2.12/${SPARK_VER}/spark-sql-kafka-0-10_2.12-${SPARK_VER}.jar \
 && curl -fsSLO https://repo1.maven.org/maven2/org/apache/spark/spark-token-provider-kafka-0-10_2.12/${SPARK_VER}/spark-token-provider-kafka-0-10_2.12-${SPARK_VER}.jar \
 && curl -fsSLO https://repo1.maven.org/maven2/org/apache/kafka/kafka-clients/${KAFKA_CLIENT}/kafka-clients-${KAFKA_CLIENT}.jar \
 && curl -fsSLO https://repo1.maven.org/maven2/org/apache/commons/commons-pool2/2.11.1/commons-pool2-2.11.1.jar
USER spark
