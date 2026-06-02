import os
from datetime import datetime
from pyspark.sql import SparkSession, DataFrame
from graphframes import GraphFrame
from pyspark.sql.functions import col, count, desc, array, lit, countDistinct, monotonically_increasing_id
from functools import reduce

if __name__ == "__main__":

    #creates spark session
    #appName is used to identify the job in the history server
    spark = (
        SparkSession.builder
        .appName("task3")
        .config("spark.jars.packages", "graphframes:graphframes:0.8.3-spark3.5-s_2.12")
        .getOrCreate()
    )

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

    #Set checkpoint dir
    checkpoint_dir = f"s3a://{s3_bucket}/spark_checkpoints"
    spark.sparkContext.setCheckpointDir(checkpoint_dir)


    airports = "s3a://" + s3_data_repository_bucket + "/ECS765/openflights/airports.csv"
    routes = "s3a://" + s3_data_repository_bucket + "/ECS765/openflights/routes.csv"

    #load csv files as DataFrames
    df_airports = spark.read.csv("s3a://" + s3_data_repository_bucket + "/ECS765/openflights/airports.csv", header=True, inferSchema=True)
    df_routes = spark.read.csv("s3a://" + s3_data_repository_bucket + "/ECS765/openflights/routes.csv", header=True, inferSchema=True)

    #removes any unnecessary spaces from all column names
    df_routes = df_routes.toDF(*[c.strip() for c in df_routes.columns])
    
    #compute raw edges and raw vertices
    vertices_raw = df_airports.count()
    edges_raw = df_routes.count()

    #select and rename airports columns and clean
    #cast numerical fields to int/double
    df_vertices = (
        df_airports
        .select(
            col("Airport ID").alias("id"),
            col("Name").alias("name"),
            col("City").alias("city"),
            col("Country").alias("country"),
            col("IATA").alias("iata"),
            col("Latitude").alias("lat"),
            col("Longitude").alias("lon")
        )
        .withColumn("id", col("id").cast("int"))
        .withColumn("lat", col("lat").cast("double"))
        .withColumn("lon", col("lon").cast("double"))
        .dropna(subset=["id", "lat", "lon"])
        .dropDuplicates(["id"])
    )
    #compute vertices count of cleaned data
    vertices_clean = df_vertices.count()

    df_routes.printSchema()
    
    #select and rename routes columns and clean
    #cast numerical fields to int/double
    df_edges = (
        df_routes
        .select(
            col("source airport id").alias("src"),
            col("destination airport id").alias("dst"),
            col("airline").alias("airline"),
            col("stops").alias("stops")
        )
        .withColumn("src", col("src").cast("int"))
        .withColumn("dst", col("dst").cast("int"))
        .withColumn("stops", col("stops").cast("int"))
        .dropna(subset=["src", "dst"])
    )

    #remove routes whose endpoints are missing in airport vertices
    df_edges_src = df_edges.join(df_vertices.select("id"), df_edges.src == df_vertices.id, "inner").drop("id")
    df_clean_edges = df_edges_src.join(df_vertices.select("id"), df_edges_src.dst == df_vertices.id, "inner").drop("id")

    df_edges_count = df_clean_edges.count()
    no_endpoint_edges = edges_raw - df_edges_count

    #create GraphFrame
    g = GraphFrame(df_vertices, df_clean_edges)

    #show tales for edges and vertices
    df_vertices.show(10, truncate=False)
    df_clean_edges.show(10, truncate=False)

    #show summary results
    summary_table = [
        ("Raw Vertices", vertices_raw),
        ("Clean Vertices", vertices_clean),
        ("Raw Edges", edges_raw),
        ("Edges Removed (no endpoint)", no_endpoint_edges),
        ("Clean Edges", df_edges_count),
    ]
    df_summary = spark.createDataFrame(summary_table, ["Metric", "Value"])
    df_summary.show(truncate=False)

    #save results to s3
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    df_vertices.coalesce(1).write.csv(f"s3a://{s3_bucket}/task3_q1_vertices_output_{now}.csv", header=True)
    df_clean_edges.coalesce(1).write.csv(f"s3a://{s3_bucket}/task3_q1_edges_output_{now}.csv", header=True)

    #*******************
    #Graph Connectivity
    #*******************

    #compute connected components
    connected_components = g.connectedComponents()

    #compute component sizes
    component_sizes = (
        connected_components
        .groupBy("component")
        .agg(count("*").alias("size"))
        .orderBy(col("size").desc())
    )
    
    #compute number of connected components
    components_number = component_sizes.count()
    print("Number of connected components: ", components_number)

    #show top5 component sizes
    top5_components = (
        component_sizes
        .select(
            col("component").alias("Component ID"),
            col("size")
        )
        .orderBy(col("size").desc())
        .limit(5)
    )
    top5_components.show(truncate=False)

    #save results to s3
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    top5_components.coalesce(1).write.csv(f"s3a://{s3_bucket}/task3_q2a_output_{now}.csv", header=True)

    #****************************
    #Degree and Triangle analysis
    #****************************

    #compute in-degree and out-degree
    in_degree = g.inDegrees
    out_degree = g.outDegrees

    #compute total degree
    df_degree = (
        in_degree.join(out_degree, on="id", how="outer")
        .na.fill(0)
        .withColumn("Total_Degree", col("inDegree") + col("outDegree"))
    )

    #use airport name for top10 table
    degree_airport = (
        df_degree
        .join(df_vertices.select("id", "name"), on="id")
    )

    #top10 airports by total degree
    df_top10_degree = (
        degree_airport
        .orderBy(desc("Total_Degree"))
        .select(
            col("name").alias("Airport"),
            col("Total_Degree").alias("Total Degree")
            
        )
        .limit(10)
    )

    #show top10 table
    df_top10_degree.show(truncate=False)

    #compute triangle count
    df_triangle = g.triangleCount()

    #use airport name for top10 table
    triangle_airport = (
        df_triangle
        .join(df_vertices.select(col("id"), col("name").alias("airport_name")), on="id")
    )

    #top10 airports by total degree
    df_top10_triangle = (
        triangle_airport
        .orderBy(desc("count"))
        .select(
            col("airport_name").alias("Airport"),
            col("count").alias("Triangle Count")
            
        )
        .limit(10)
    )

    #show top10 table
    df_top10_triangle.show(truncate=False)

    #save results to s3
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    df_top10_degree.coalesce(1).write.csv(f"s3a://{s3_bucket}/task3_q2b_degree_{now}.csv", header=True)
    df_top10_triangle.coalesce(1).write.csv(f"s3a://{s3_bucket}/task3_q2b_triangle_{now}.csv", header=True)

    #******************************************
    #Shortest Flight Paths & Route Reachability
    #******************************************

    #start and goal airport iata
    start_airport = "LHR"
    goal_airport = "SYD"

    #select start airport using ID
    df_start_id = (
        df_vertices
        .filter(col("iata") == start_airport)
        .select("id")
        .first()[0]
    )

    #select goal airport using ID
    df_goal_id = (
        df_vertices
        .filter(col("iata") == goal_airport)
        .select("id")
        .first()[0]
    )

    #run Breadth-First Search up to depth four
    breadth_first_search = g.bfs(
        fromExpr=f"id = {df_start_id}",
        toExpr=f"id = {df_goal_id}",
        maxPathLength = 4
    )

    #show bfs result path
    breadth_first_search.show(truncate=False)

    #choose a path
    bfs_path = breadth_first_search.limit(1)

    #choose hops for table
    hops = []

    hops.append(
        bfs_path.select(col("from.id").alias("id")).withColumn("Hop", lit(1))
    )
    hop_counter = 2
    
    if "v1" in bfs_path.columns:
        hops.append(
            bfs_path.select(col("v1.id").alias("id")).withColumn("Hop", lit(hop_counter))
        )
        hop_counter += 1
        
    if "v2" in bfs_path.columns:
        hops.append(
            bfs_path.select(col("v2.id").alias("id")).withColumn("Hop", lit(hop_counter))
        )
        hop_counter+=1
        
    if "v3" in bfs_path.columns:
        hops.append(
            bfs_path.select(col("v3.id").alias("id")).withColumn("Hop", lit(hop_counter))
        )
        hop_counter+=1
        
    #final destinantion hop
    hops.append(
        bfs_path.select(col("to.id").alias("id")).withColumn("Hop", lit(hop_counter))
    )
    
    #combine hops and create table
    hops_table = reduce(DataFrame.unionAll, hops).dropna().distinct()

    hops_summary = (
        hops_table
        .join(df_vertices.select("id", "name", "city", "country"), on="id")
        .select("Hop", col("name").alias("Airport"), "city", "country")
        .orderBy("Hop")
    )

    #show the table in the output
    hops_summary.show(truncate=False)

    #shortest path reachability
    shortest_paths = g.shortestPaths(
        landmarks = [df_goal_id]
    )
    shortest_paths_result = shortest_paths.limit(10)
    shortest_paths_result.show(truncate=False)

    #*****************
    #Classic Page Rank
    #*****************

    #run page rank with resetProbability of 0.15

    page_rank = g.pageRank(resetProbability = 0.15, maxIter = 20)

    #extract vertices pageRank scores
    vertices_page_rank = page_rank.vertices

    #join with airport info for table readability
    page_rank_airports = (
        vertices_page_rank
        .join(df_vertices.select(col("id"), col("name").alias("airport_name")), on="id")
    )

    #rank top10 airport by descending pageRank
    page_rank_top10 = (
        page_rank_airports
        .orderBy(col("pagerank").desc())
        .select(
            col("airport_name").alias("Airport"),
            col("pagerank").alias("PageRank Score")
        )
        .limit(10)
    )

    #show table
    page_rank_top10.show(truncate=False)

    #save results to s3
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    page_rank_top10.coalesce(1).write.csv(f"s3a://{s3_bucket}/task3_q4a_output_{now}.csv", header=True)

    #**************************************
    #Weighted PageRank by Airline Diversity
    #**************************************

    #compute airline route diversity per airport
    airline_diversity = (
        df_clean_edges
        .groupBy("src")
        .agg(countDistinct("airline").alias("weight"))
        .withColumnRenamed("src", "id")
    )

    #build weighted edges using weight
    weighted_edges = (
        df_clean_edges
        .join(
            airline_diversity,
            df_clean_edges.src == airline_diversity.id,
            how="left"
        )
        .drop("id")
    )

    #build weighted graphframe
    weighted_graph = GraphFrame(df_vertices, weighted_edges)

    #run weighted pageRank
    weighted_pr = weighted_graph.pageRank(
        resetProbability = 0.15,
        maxIter = 20,
        #edgeWeightCol = "weight"
    )

    weighted_pr_vertices = weighted_pr.vertices

    #weighted pageRank top10 table by descending pageRank
    top10_weighted = (
        weighted_pr_vertices
        .join(df_vertices.select(col("id"), col("name").alias("airport_name")), on="id")
        .orderBy(col("pagerank").desc())
        .select(
            col("airport_name").alias("Airport"),
            col("pagerank").alias("Weighted PageRank Score")
        )
        .limit(10)
    )

    #comparison table for classic and weighted PageRank

    #rename classic ranked columns
    rank_classic = (
        page_rank_top10
        .withColumnRenamed("Airport", "Classic Top Airport")
        .withColumnRenamed("PageRank Score", "PR")
        .withColumn("Rank", monotonically_increasing_id() + 1)
    )
    
    #rename weighted ranked columns
    rank_weighted = (
        top10_weighted
        .withColumnRenamed("Airport", "Weighted Top Airport")
        .withColumnRenamed("Weighted PageRank Score", "W-PR")
        .withColumn("Rank", monotonically_increasing_id() + 1)
    )

    #comparison of classic and ranked
    comparison_table = (
        rank_classic
        .join(rank_weighted, on="Rank")
        .select("Rank", "Classic Top Airport", "PR", "Weighted Top Airport", "W-PR")
    )

    #show comparison table
    comparison_table.show(truncate=False)

    #save results to s3
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_table.coalesce(1).write.csv(f"s3a://{s3_bucket}/task3_q4b_output_{now}.csv", header=True)


    #**************************************
    #Community Detection(Label Propagation)
    #**************************************

    #run label propagation to apply community detection algorithm
    communities = g.labelPropagation(maxIter=10)

    #join results with airport metadata
    airport_communities = (
        communities
        .join(
            df_vertices.select(col("id"), col("name").alias("airport_name"), col("city").alias("airport_city"), col("country").alias("airport_country")),
            on="id"
        )
    )

    #compute number of detected communities
    detected_communities = communities.select("label").distinct().count()
    print("Number of Detected Communities: ", detected_communities)

    #find top5 largest communities by size
    community_sizes = (
        communities
        .groupBy("label")
        .count()
        .withColumnRenamed("label", "Community ID")
        .withColumnRenamed("count", "Size")
        .orderBy(col("Size").desc())
    )
    top5_communities = community_sizes.limit(5)
    top5_communities.show(truncate=False)

    #map communities to geographic clusters
    community_mapping = (
        airport_communities
        .groupBy("label", "airport_country")
        .count()
        .orderBy(col("label"), col("count").desc())
    ).limit(20)

    #show table of the mapping
    community_mapping.show(truncate=False)

    #choose a large community
    large_community = top5_communities.first()["Community ID"]

    #find top hubs inside largest community
    community_top_hubs = (
        g.degrees
        .join(
            communities.filter(col("label") == large_community),
            on="id"
        )
        .join(
            df_vertices.select(col("id"), col("name").alias("airport_name"), col("city").alias("airport_city"), col("country").alias("airport_country")),
            on="id"
        )
        .orderBy(col("degree").desc())
        .select("airport_name", "degree")
        .limit(10)
    )
    
    community_top_hubs.show(truncate=False)

    #save results to s3
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    top5_communities.coalesce(1).write.csv(f"s3a://{s3_bucket}/task3_q5_top5_{now}.csv", header=True)
    community_top_hubs.coalesce(1).write.csv(f"s3a://{s3_bucket}/task3_q5_hubs_{now}.csv", header=True)
    
    spark.stop()



