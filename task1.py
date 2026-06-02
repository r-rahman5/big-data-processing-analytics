import os
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp, date_format, count, hour, dayofweek, when, countDistinct, row_number
from pyspark.sql.functions import sum as spark_sum, when
from pyspark.sql.window import Window

if __name__ == "__main__":
    
    #creates spark session
    #appName is used to identify the job in the history server
    spark = SparkSession.builder.appName("task1").getOrCreate()

    s3_data_repository_bucket = os.environ['DATA_REPOSITORY_BUCKET']
    s3_endpoint_url = os.environ['S3_ENDPOINT_URL'] + ':' + os.environ['BUCKET_PORT']
    s3_access_key_id = os.environ['AWS_ACCESS_KEY_ID']
    s3_secret_access_key = os.environ['AWS_SECRET_ACCESS_KEY']
    s3_bucket = os.environ['BUCKET_NAME']

    hadoopConf = spark.sparkContext._jsc.hadoopConfiguration()
    hadoopConf.set("fs.s3a.endpoint", s3_endpoint_url)
    hadoopConf.set("fs.s3a.access.key", s3_access_key_id)
    hadoopConf.set("fs.s3a.secret.key", s3_secret_access_key)
    hadoopConf.set("fs.s3a.path.style.access", "true")
    hadoopConf.set("fs.s3a.connection.ssl.enabled", "false")

    #loads GitHub archive data for 2024 into a single data frame
    df = spark.read.json("s3a://" + s3_data_repository_bucket + "/ECS765/gharchive/2024-10/*.json.gz")

    #converts "created_at" from string to a Spark timestamp
    #derive the summary metrics(date, hour, dow)
    df = df.withColumn("created_at_ts", to_timestamp(col("created_at"))) \
           .withColumn("date", date_format(col("created_at_ts"), "yyyy-MM-dd")) \
           .withColumn("hour", hour(col("created_at_ts"))) \
           .withColumn("dow", dayofweek(col("created_at_ts")))

    #the Dataframe is cached in memory to avoid unnecessary recomputations
    df.cache()
    row_count = df.count()

    df.printSchema()
    print("Row count: ", row_count)
    #show 10 row preview
    df.show(10, truncate=False)

    #aggregation that checks if date extraction functions correctly
    events_per_day = df.groupBy("date").agg(count("*").alias("events"))
    events_per_day.show(5)

    #save results to s3 bucket
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    events_per_day.coalesce(1).write.csv(f"s3a://{s3_bucket}/task1_q1_output_{now}.csv", header=True)

    #print summary metrics for q1
    print("Distinct event types: ", df.select("type").distinct().count())
    print("Distinct repositories: ", df.select("repo.id").distinct().count())
    print("Distinct actors: ", df.select("actor.id").distinct().count())

    #******************
    #weekday vs weekend
    #******************
    
    #group the events by dow and compute total events
    df_q2 = df.groupBy("dow") \
           .agg(count("*").alias("event_count"))
    
    #add column to show the day of the week of the events
    df_q2 = df_q2.withColumn(
        "Day of Week",
        when(col("dow") == 1, "Sunday")
        .when(col("dow") == 2, "Monday")
        .when(col("dow") == 3, "Tuesday")
        .when(col("dow") == 4, "Wednesday")
        .when(col("dow") == 5, "Thursday")
        .when(col("dow") == 6, "Friday")
        .when(col("dow") == 7, "Saturday")
    )
    
    #fix ordering of days to show Monday -> Sunday
    df_q2 = df_q2.orderBy(
        when(col("dow") == 2, 1)
        .when(col("dow") == 3, 2)
        .when(col("dow") == 4, 3)
        .when(col("dow") == 5, 4)
        .when(col("dow") == 6, 5)
        .when(col("dow") == 7, 6)
        .when(col("dow") == 1, 7)
    )

    #show table
    df_q2.show(truncate=False)

    #Save to csv
    df_q2.coalesce(1).write.csv(f"s3a://{s3_bucket}/task1_q2_output_{now}.csv", header=True)

    #*********************
    #Repository Popularity
    #*********************

    #filter for only WatchEvent and PullRequestEvent
    df_q3a = df.filter((col("type") == "WatchEvent") | (col("type") == "PullRequestEvent"))

    #group by repo full name, type and count events
    df_grp_q3a = df_q3a.groupBy("type", "repo.name") \
                       .agg(count("*").alias("event_count"))

    #sort by descending event count and get top 10
    df_top10_q3a = df_grp_q3a.orderBy(col("event_count").desc()).limit(10)

    #show the top 10
    df_top10_q3a.show(truncate=False)
    
    #save results to s3 bucket
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    df_top10_q3a.coalesce(1).write.csv(f"s3a://{s3_bucket}/task1_q3a_output_{now}.csv", header=True)

    #*********************
    #Actor Diversity
    #*********************

    #for each event type count actors per repo
    df_q3b = df_q3a.groupBy("type", "repo.name") \
                   .agg(countDistinct("actor.id").alias("distinct_actors"),
                        count("*").alias("event_count"))
    
    #sort distinct actors in descending order
    df_top10_q3b = df_q3b.orderBy(col("distinct_actors").desc()).limit(10)

    #show the top 10
    df_top10_q3b.show(truncate=False)

    #save results to s3 bucket
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    df_top10_q3b.coalesce(1).write.csv(f"s3a://{s3_bucket}/task1_q3b_output_{now}.csv", header=True)


    #****************
    #New Contributors
    #****************

    #define a window: for each actor ordered by a timestamp
    actor_w = Window.partitionBy("actor.id").orderBy(col("created_at_ts"))

    #then can assign an event to a row number
    df_row = df.withColumn("row_num", row_number().over(actor_w))

    #filter for row number == 1
    df_first_event = df_row.filter(col("row_num") == 1)

    #group by repo full name and count distinct new actors
    df_q4 = df_first_event.groupBy("repo.name") \
        .agg(countDistinct("actor.id").alias("new_contributors"))

    #show top-10 repos
    df_top10_q4 = df_q4.orderBy(col("new_contributors").desc()).limit(10)
    df_top10_q4.show(truncate=False)

    #save results to s3 bucket
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    df_top10_q4.coalesce(1).write.csv(f"s3a://{s3_bucket}/task1_q4_output_{now}.csv", header=True)


    #************************
    #Time-of-Day Productivity
    #************************

    #filter PushEvent 
    df_q5 = df.filter(col("type") == "PushEvent")

    #extract commit count from payload.size
    df_q5 = df_q5.withColumn("commit_count", col("payload.size").cast("int"))

    #group by hour and sum commits
    df_hourly_q5 = df_q5.groupBy("hour") \
        .agg(spark_sum("commit_count").alias("total_commits")) \
        .orderBy("hour")
    
    df_hourly_q5.show(24)

    #save results to s3 bucket
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    df_hourly_q5.coalesce(1).write.csv(f"s3a://{s3_bucket}/task1_q5_output_{now}.csv", header=True)

    #********************************
    #Organisational Activity Analysis
    #********************************

    #filter non-null organisations
    df_q6 = df.filter(col("org.login").isNotNull())

    #group by org.login and count events and list top-5
    df_group_org = df_q6.groupBy("org.login") \
        .agg(count("*").alias("event_count")) \
        .orderBy(col("event_count").desc()) \
        .limit(5)
    df_group_org.show(truncate=False)

    #save results to s3 bucket
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    df_group_org.coalesce(1).write.csv(f"s3a://{s3_bucket}/task1_q6_output_{now}.csv", header=True)
    
    spark.stop()

