import os
import sys
import time
import pandas as pd
import random
#os.environ["SPARK_HOME"] = "/usr/local/spark"
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
df = pd.DataFrame(columns=['src', 'dst', 'flight number'])
flightno=[]
station={}

for i in range(201):
    df.at[i,"flight number"]=(random.randint(1111,9999))
    station=random.sample(airports, 2)
    df.at[i,"src"]=station[0]
    df.at[i,"dst"]=station[1]
print(df)
sqlContext = SQLContext(sc)
e =sqlContext.createDataFrame(df)
v = sqlContext.createDataFrame([("a","BHM"),("b","DHN"),("c","HSV"),("d","MGM"),("e","MRI"),("f","BWI"),("g","LAX"),("h","JFK"),("g","MIA"),("i","PHL")],["id", "name"] )
g = GraphFrame(v, e)
inDegreeDF=g.inDegrees
outDegreeDF=g.outDegrees
degreeDF=g.degrees
inDegreeDF.sort(['inDegree'],ascending=[0]).show()
outDegreeDF.sort(['outDegree'],ascending=[0]).show()
degreeDF.show()
