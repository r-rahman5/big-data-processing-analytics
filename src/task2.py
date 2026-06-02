import os
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp, date_format, month, count, hour, countDistinct, min, max, to_date, desc, row_number, concat, lit, desc, when, mean, stddev, expr
from pyspark.sql.window import Window
from pyspark.sql.functions import abs as spark_abs

if __name__ == "__main__":
    
    #creates spark session
    #appName is used to identify the job in the history server
    spark = SparkSession.builder.appName("task2").getOrCreate()

    #configure s3
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

    #load csv files into a single DataFrame
    df = spark.read.csv("s3a://" + s3_data_repository_bucket + "/ECS765/tfl/cyclehire/2024-q3/*.csv", header=True, inferSchema=True)
    
    #parse timestamps to obtain date, hour, month and ensure duration is an integer
    df = (
        df
        .withColumn("StartTime", to_timestamp(col("Start date"), "dd/MM/yyyy HH:mm"))
        .withColumn("month", month(col("StartTime")))
        .withColumn("month_name", date_format(col("StartTime"), "MMM"))
        .withColumn("date", to_date(col("StartTime")))
        .withColumn("hour", hour(col("StartTime")))
        .withColumn("Duration", (col("Total duration (ms)").cast("long")/1000).cast("int"))
    )

    #remove records where ID is missing
    #remove any negative or null values
    df_q1 = (
        df
        .filter(col("Start station number").isNotNull())
        .filter(col("End station number").isNotNull())
        .filter(col("Duration").isNotNull())
        .filter(col("Duration") > 0)
        .filter(col("StartTime").isNotNull())
    )

    #cache the cleaned DataFrame
    df_q1.cache()
    #print cleaned Row Count
    print("Row Count: ", df_q1.count())

    #print the schema
    print("Schema: ")
    df_q1.printSchema()


    #summary statistics
    df_summary_q1 = df_q1.agg(
        count("*").alias("Total Rows"),
        countDistinct("Start station number").alias("Distinct Start Stations"),
        countDistinct("End station number").alias("Distinct End Stations"),
        min("Duration").alias("Min Duration (in s)"),
        max("Duration").alias("Max Duration (in s)"),
        min("date").alias("Start Date"),
        max("date").alias("End Date")
        
    )

    #print the summary statistics
    print("Summary Statistics: ")
    df_summary_q1.show(truncate=False)

    #trips_per_month = df.groupBy("month").agg(count("*").alias("trip_count"))
    #trips_per_month.show(5)

    #save results to s3 bucket
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    df_summary_q1.coalesce(1).write.csv(f"s3a://{s3_bucket}/task2_q1_output_{now}.csv", header=True)

    #****************************
    #Station Popularity
    #****************************

    k = 10 #chosen k-value

    #overall top-K start stations by trip count
    q2a_topk = (
        df_q1
        .groupBy("Start station number", "Start station")
        .agg(count("*").alias("Trip Starts"))
        .orderBy(desc("Trip Starts"))
        .limit(k)
    )

    #monthly trip starts count
    q2a_monthly_counts = (
        df_q1
        .groupBy("month","month_name", "Start station number", "Start station")
        .agg(count("*").alias("Trip Starts"))
    )

    #rank stations each month using windows
    w = Window.partitionBy("month").orderBy(col("Trip Starts").desc())

    #monthly top-K stations
    q2a_monthly_topk = (
        q2a_monthly_counts
        .withColumn("Rank", row_number().over(w))
        .filter(col("Rank") <= k)
        .orderBy("month", "Rank")
    )
    
    #show each table
    q2a_topk.show(truncate=False)
    q2a_monthly_topk.show(truncate=False)

    #save results to s3 bucket
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    q2a_topk.coalesce(1).write.csv(f"s3a://{s3_bucket}/task2_q2a_overall_output_{now}.csv", header=True)
    q2a_monthly_topk.coalesce(1).write.csv(f"s3a://{s3_bucket}/task2_q2a_monthly_output_{now}.csv", header=True)

    #***************
    #Route Flows
    #***************

    #construct the route
    df_q2b = (
        df_q1
        .withColumn(
            "Route",
            concat(col("Start station"), lit(" -> "), col("End station"))
        )
    )

    #filter out self-loop routes
    df_q2b = df_q2b.filter(col("Start station") != col("End station"))

    #compute the 20 most frequent routes
    df_freq_routes = (
        df_q2b
        .groupBy("Route")
        .agg(count("*").alias("Trips"))
        .orderBy(col("Trips").desc())
        .limit(20)
    )

    #to find top10 routes for the table and show
    df_top10 = df_freq_routes.limit(10)
    df_top10.show(truncate=False)

    #save results to s3 bucket
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    df_freq_routes.coalesce(1).write.csv(f"s3a://{s3_bucket}/task2_q2b_output_{now}.csv", header=True)

    #**************************
    #Utilisation By Time of Day
    #**************************

    #create hour-based time off day buckets
    df_q3 = (
        df_q1
        .withColumn(
            "time_of_day",
            when((col("hour") >= 5) & (col("hour") <= 11), "Morning")
            .when((col("hour") >= 12) & (col("hour") <= 16), "Afternoon")
            .when((col("hour") >= 17) & (col("hour") <= 21), "Evening")
            .otherwise("Night")
        )
    )

    #aggregate statistics for each bucket
    df_q3_stats = (
        df_q3.groupBy("time_of_day")
        .agg(
            count("*").alias("Trips"),
            expr("percentile_approx(Duration, 0.5)").alias("Median (s)"),
            expr("percentile_approx(Duration, 0.9)").alias("90th Percentile (s)")
        )
    )

    #sort results in order by time of day
    df_q3_order = (
        df_q3_stats
        .withColumn(
            "order",
            when(col("time_of_day")=="Morning", 1)
            .when(col("time_of_day")=="Afternoon", 2)
            .when(col("time_of_day")=="Evening", 3)
            .when(col("time_of_day")=="Night", 4)
        )
        .orderBy("order")
        .drop("order")
    )
    #show table
    df_q3_order.show(truncate=False)
    
    #save results to s3 bucket
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    df_q3_order.coalesce(1).write.csv(f"s3a://{s3_bucket}/task2_q3_output_{now}.csv", header=True)

    #**************************
    #Daily Anomaly Scan
    #**************************

    #filter data to obtain August 2024
    df_q4 = df_q1.filter(col("month")==8)

    #compute daily mean duration and total trip count
    df_q4_stats = (
        df_q4.groupBy("date")
        .agg(
            mean("Duration").alias("Avg_Duration_(s)"),
            count("*").alias("Trip_Count")
        )
    )

    #compute mean and standard deviation
    df_vals = df_q4_stats.agg(
        mean("Avg_Duration_(s)").alias("mean_dur"),
        mean("Trip_Count").alias("mean_trip"),
        stddev("Avg_Duration_(s)").alias("stdv_dur"),
        stddev("Trip_Count").alias("stdv_trip")
    ).collect()[0]

    #assign values so z-score can be computed
    mean_duration = df_vals["mean_dur"]
    mean_trip_count = df_vals["mean_trip"]
    std_duration = df_vals["stdv_dur"]
    std_trip_count = df_vals["stdv_trip"]

    #compute z-score and flag any anomalies using boolean field
    df_q4_zscores = (
        df_q4_stats
        .withColumn(
            "zscore_duration",
            (col("Avg_Duration_(s)") - mean_duration) / std_duration
        )
        .withColumn(
        "zscore_trip_count",
        (col("Trip_Count") - mean_trip_count) / std_trip_count
        )
        .withColumn(
            "is_anomaly",
            (col("zscore_duration") > 2) | (col("zscore_trip_count") < -2)
        )
        .orderBy("date")
    )
    #show anomalies
    #df_q4_anomaly.filter(col("is_anomaly") == True).show(truncate=False)
    df_q4_zscores.show(31, truncate=False)
    

    #save results to s3 bucket
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    df_q4_zscores.coalesce(1).write.csv(f"s3a://{s3_bucket}/task2_q4_output_{now}.csv", header=True)


    #**************************
    #Peak Hour Inflow/Outflow
    #**************************

    #define peak windows and keep only peak hours
    df_q5_peak = (
        df_q1
        .withColumn(
            "peak_window",
            when((col("hour") >= 7) & (col("hour") <= 10), "Morning")
            .when((col("hour") >= 16) & (col("hour") <= 19), "Evening")
        )
        .filter(col("peak_window").isNotNull())
    )

    #for each station in each peak window, count starts
    df_q5_starts = (
        df_q5_peak
        .groupBy("peak_window", "Start station")
        .agg(count("*").alias("Starts"))
        .withColumnRenamed("Start station", "Station Name")
    )
    
    #for each station in each peak window, count ends
    df_q5_ends = (
        df_q5_peak
        .groupBy("peak_window", "End station")
        .agg(count("*").alias("Ends"))
        .withColumnRenamed("End station", "Station Name")
    )

    #join starts and ends to compute the netflow(net_flow = starts - ends)
    df_q5_net_flow = (
        df_q5_starts
        .join(
            df_q5_ends,
            ["peak_window", "Station Name"],
            how = "outer"
        )
        .fillna(0)
        .withColumn("Net Flow", col("Starts") - col("Ends"))
    )

    #rank by absoloute net flow in each peak window
    w = Window.partitionBy("peak_window").orderBy(spark_abs(col("Net Flow")).desc())
    df_q5_ranked = df_q5_net_flow.withColumn(
        "Rank", row_number().over(w)
    )

    #top-10 highes net inflow
    df_q5_inflow = df_q5_ranked.filter((col("Net Flow") > 0) & (col("Rank") <= 10))

    #top-10 highes net outflow
    df_q5_outflow = df_q5_ranked.filter((col("Net Flow") < 0) & (col("Rank") <= 10))

    #show results
    df_q5_inflow.show(truncate=False)
    df_q5_outflow.show(truncate=False)

    #save results to s3 bucket
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    df_q5_ranked.coalesce(1).write.csv(f"s3a://{s3_bucket}/task2_q5_output_{now}.csv", header=True)
    
    spark.stop()

