import os
import sys
import sqlite3
os.environ["SPARK_HOME"] = "/usr/local/spark"
os.environ['SPARK_HOME']="C:\spark-3.0.0-preview2-bin-hadoop2.7"

# Append pyspark  to Python Path
sys.path.append("C:\spark-3.0.0-preview2-bin-hadoop2.7\python\pyspark")
from pyspark import SparkConf, SparkContext
conf=SparkConf().setMaster("local").setAppName("Browsing history")
sc= SparkContext(conf=conf)
data=[("Maryland", 2), ("Virginia", 5), ("Delaware", 1),("Maryland", 3), ("Delaware", 1) ]
rdd = sc.parallelize( data,2 )

sumCount = rdd.combineByKey(lambda value: (value, 1),
                            lambda x, value: (x[0] + value, x[1] + 1),
                            lambda x, y: (x[0] + y[0], x[1] + y[1])
                           )
averageByKey = sumCount.map(lambda label_value_sum_count: (label_value_sum_count[0], label_value_sum_count[1][0] / label_value_sum_count[1][1]))
print(averageByKey.collectAsMap())
