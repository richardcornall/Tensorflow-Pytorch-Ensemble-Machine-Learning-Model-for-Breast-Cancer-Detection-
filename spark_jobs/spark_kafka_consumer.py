from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StructField, StringType, FloatType

if __name__ == '__main__':
    spark = SparkSession \
        .builder \
        .appName("KafkaSparkConsumer") \
        .getOrCreate()

    # Define schema of the incoming data
    schema = StructType([
        StructField("id", StringType()),
        StructField("diagnosis", StringType()),
        # Add other fields as per your dataset
        StructField("radius_mean", FloatType()),
        StructField("texture_mean", FloatType()),
        # ... include all necessary fields
    ])

    # Read data from Kafka
    df = spark \
        .readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "localhost:9092") \
        .option("subscribe", "breast_cancer_data") \
        .load()

    # Deserialize JSON data
    json_df = df.selectExpr("CAST(value AS STRING) as json_string")
    data_df = json_df.select(from_json(col("json_string"), schema).alias("data")).select("data.*")

    # Perform transformations or write to sink
    query = data_df.writeStream \
        .outputMode("append") \
        .format("console") \
        .start()

    query.awaitTermination()
