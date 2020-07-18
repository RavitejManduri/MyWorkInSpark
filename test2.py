import os
import sys
import time
import pandas as pd
import random
os.environ["SPARK_HOME"] = "/usr/local/spark"
os.environ['SPARK_HOME']="C:\spark-3.0.0-preview2-bin-hadoop2.7"
# Append pyspark  to Python Path
sys.path.append("C:\spark-3.0.0-preview2-bin-hadoop2.7\python\pyspark")
from pyspark import SparkConf, SparkContext
conf=SparkConf().setMaster("local").setAppName("Graphx")
sc = SparkContext(conf=conf)
#airports=["BHM","DHN","HSV","MGM","MRI","BWI","LAX","JFK","MIA","PHL"]
airports=["a","b","c","d","e","f","g","h","i"]
from pyspark.sql import SQLContext
from graphframes import *
from graphframes import *
from pyspark.sql import SQLContext
sqlContext = SQLContext(sc)

# Create a Vertex DataFrame with unique ID column "id"
v = sqlContext.createDataFrame([
  ("a", "Alice", 34),
  ("b", "Bob", 36),
  ("c", "Charlie", 30),
], ["id", "name", "age"])

# Create an Edge DataFrame with "src" and "dst" columns
e = sqlContext.createDataFrame([
  ("a", "b", "friend"),
  ("b", "c", "follow"),
  ("c", "b", "follow"),
], ["src", "dst", "relationship"])
# Create a GraphFrame
g = GraphFrame(v, e)

# Query: Get in-degree of each vertex.
g.inDegrees.show()

# Query: Count the number of "follow" connections in the graph.
g.edges.filter("relationship = 'follow'").count()

# Run PageRank algorithm, and show results.
results = g.pageRank(resetProbability=0.01, maxIter=20)
results.vertices.select("id", "pagerank").show()
