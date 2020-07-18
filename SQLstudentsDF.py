 import os
import sys
import sqlite3
os.environ["SPARK_HOME"] = "/usr/local/spark"
os.environ['SPARK_HOME']="C:\spark-3.0.0-preview2-bin-hadoop2.7"
# Append pyspark  to Python Path
sys.path.append("C:\spark-3.0.0-preview2-bin-hadoop2.7\python\pyspark")
from pyspark import SparkConf, SparkContext
conf=SparkConf().setMaster("local").setAppName("SQL Grade")
sc = SparkContext(conf=conf)
from pyspark.sql import SQLContext
from pyspark.sql.types import *

sqlContext = SQLContext(sc)
from pyspark.sql import Row
#•	Grade A is for GPA 3.6 or higher, Grade B for GPA 3.2 to less than 3.6, Grade C for GPA 2.8 to less than 3.2.
def grade(x):
    if(x>=3.6):
        return 'A'
    elif(x>=3.2):
        return 'B'
    elif(x>=2.8):
        return 'C'
    else:
        return 'D'
# Load a text file and convert each line to a Row.
lines = sc.textFile(r"C:\Users\ravit\Documents\data.txt")
parts = lines.map(lambda l: l.split(","))
peopleRDD = parts.map(lambda p: Row(StudentID=int(p[0]), StudentName=p[1], PhoneNumber=str(p[2]),Grade=str(grade(float(p[3])))))

# Infer the schema, and register the DataFrame as a table.
schemaPeople = sqlContext.createDataFrame(peopleRDD)
schemaPeople.createOrReplaceTempView("student")
teenagers = sqlContext.sql("SELECT StudentID,StudentName,PhoneNumber,Grade FROM student")
print(teenagers.show())

