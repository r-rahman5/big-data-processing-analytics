import sys, string
import os
import socket
import time
import operator
import boto3
import json
# from pyspark.sql import SparkSession
# from pyspark.streaming import StreamingContext

from pyspark.sql import Row, SparkSession
from pyspark.sql.streaming import DataStreamWriter, DataStreamReader
from pyspark.sql.functions import explode ,split ,window, when, current_timestamp
from pyspark.sql.types import IntegerType, DateType, StringType, StructType
from pyspark.sql.functions import sum,avg,max,col,to_timestamp,regexp_extract,count,desc

#*********************************************************
#This script contains the solutions for task 4
#As each question has its own independent streaming query
#I can only execute one question at a time
#************************How to run:**********************
#keep the initaial parsing from q1a, ensure to keep both
#the regex patterns and the field extractions
#to run q1b - q5 comment out the query from q1a
#For q2-q5 you also include the malformed handling part in q1b
#as is_malformed field is referenced in q2-q5
#however ensure to comment out the rest of q1b
#then you can simply uncomment the required question and run
#*********************************************************

if __name__ == "__main__":

    spark = SparkSession\
        .builder\
        .appName("HDFSSparkStreaming")\
        .getOrCreate() \

    spark.sparkContext.setLogLevel("ERROR")

    
    #Set up the `logsDF` readStream to take in data from a socket stream
    logsDF = spark.readStream.format("socket").option("host", os.environ['STREAMING_SERVER_HDFS_2K'])\
             .option("port", int(os.environ['STREAMING_SERVER_HDFS_2K_PORT'])).load()
    

    #********************************************
    #Q1a) Streaming Ingestion and Initial Parsing
    #********************************************
    
    #regex patterns to extract fields
    timestamp_pattern = r"(\d{6}\s\d{6})"
    log_level_pattern = r"\s(INFO|WARN|ERROR)\s"
    component_pattern = r"\bdfs\.(\w+)"
    host_pattern = r"(\d{1,3}(?:\.\d{1,3}){3})(?::\d+)?"
    message_pattern = r":\s(.*)$"
    

    #derive fields using regexp
    parsed_df = (
        logsDF
        .withColumn("timestamp", to_timestamp(regexp_extract(col("value"), timestamp_pattern, 1), "yyMMdd HHmmss"))
        .withColumn("level", regexp_extract(col("value"), log_level_pattern, 1))
        .withColumn("component", regexp_extract(col("value"), component_pattern, 1))
        .withColumn("host", regexp_extract(col("value"), host_pattern, 1))
        .withColumn("message", regexp_extract(col("value"), message_pattern, 1))
    )

    
    #print schema
    parsed_df.printSchema()

    #write logs to console(append mode, no truncated columns)
    query = (
        parsed_df
        .writeStream
        .format("console")
        .option("truncate", False)
        .outputMode("append")
        .start()
    )

    query.awaitTermination()

    #**************************************************
    #Q1b) Schema enforcement and Malformed log handling
    #**************************************************

    
    #derive fields using regexp
    parsed_df = (
        logsDF
        .withColumn("timestamp", to_timestamp(regexp_extract(col("value"), timestamp_pattern, 1), "yyMMdd HHmmss"))
        .withColumn("level", regexp_extract(col("value"), log_level_pattern, 1))
        .withColumn("component", regexp_extract(col("value"), component_pattern, 1))
        .withColumn("host", regexp_extract(col("value"), host_pattern, 1))
        .withColumn("message", regexp_extract(col("value"), message_pattern, 1))
    )
    
    #include malformed handling using boolean field is_malformed
    parsed_df = (
        parsed_df
        .withColumn(
            "is_malformed",
            when(
                col("timestamp").isNull() | col("level").isNull(), True
            ).otherwise(False)
        )
    )



    #print updated schema
    parsed_df.printSchema()
    
    prev_df = (
        parsed_df
        .select("timestamp", "level", "host", "message", "is_malformed")
    )
    #write logs to console(append mode, no truncated columns)
    query = (
        prev_df
        .writeStream
        .format("console")
        .option("truncate", False)
        .outputMode("append")
        .start()
    )

    query.awaitTermination()
    #***********************************
    #Q2)watermark and late data handling
    #***********************************
    

    #applying watermark
    watermark_df = (
        parsed_df
        .filter(col("is_malformed")==False)
        .withWatermark("timestamp", "5 seconds")
        .groupBy(window(col("timestamp"), "10 seconds"), "level")
        .count()
    )
    
    #console output
    query = (
        watermark_df
        .writeStream
        .format("console")
        .outputMode("update")
        .option("truncate", "false")
        .start()
    )

    # run the query untill terminated
    query.awaitTermination()

    #*********************************
    #Q3a) Sliding Window Analysis
    #*********************************

    #sliding event time window
    sliding_df = (
        parsed_df.filter(col("is_malformed")==False)
        .filter(col("component").contains("DataNode"))
        .groupBy(window(col("timestamp"), "60 seconds", "30 seconds"))
        .count()
    )
    #console output
    query = (
        sliding_df
        .writeStream
        .format("console")
        .outputMode("update")
        .option("truncate", "false")
        .start()
    )
    #run the query untill terminated
    query.awaitTermination()
    
    #*********************************
    #Q3b) Stateful Counting
    #*********************************

    #Host state aggregation
    host_df = (
        parsed_df
        .filter(col("is_malformed")==False)
        .filter(col("host") != "")
        .withWatermark("timestamp", "60 seconds")
        .groupBy("host")
        .count()
        .orderBy(desc("count"))
    )
    #console output
    query = (
        host_df
        .writeStream
        .format("console")
        .outputMode("complete")
        .option("truncate", "false")
        .start()
    )

    #run the query untill terminated
    query.awaitTermination()
    
    #**************************************
    #Q4)Triggered Filter and Block Activity
    #**************************************

    #filter logs for messages containing a block ID
    #group by host for per-host block activity
    block_df = (
        parsed_df
        .filter(col("is_malformed")==False)
        .filter(col("level")=="INFO")
        .filter(col("message").contains("blk_"))
        .groupBy("host")
        .count()
        .withColumn("timestamp", current_timestamp())
        .orderBy(desc("count"))
    )
    #console output
    query = (
        block_df
        .writeStream
        .format("console")
        .outputMode("complete")
        .trigger(processingTime="15 seconds")
        .option("truncate", "false")
        .start()
    )

    #run the query untill terminated
    query.awaitTermination()
    
    #*******************************
    #Q5)Fault Tolerance and Recovery
    #*******************************

    #group event count per host
    df_host = (
        parsed_df
        .filter(col("is_malformed")==False)
        .filter(col("host") != "")
        .groupBy("host")
        .count()
        .orderBy(desc("count"))
    )

    #console output including checkpoint configuration
    query = (
        df_host
        .writeStream
        .format("console")
        .outputMode("complete")
        .option("checkpointLocation", f"s3a://{os.environ['BUCKET_NAME']}/checkpoints/task4")
        .option("truncate", "false")
        .start()
    )

    #run the query untill terminated
    query.awaitTermination()
    
