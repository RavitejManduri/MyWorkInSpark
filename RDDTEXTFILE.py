import os
import sys
import sqlite3
os.environ["SPARK_HOME"] = "/usr/local/spark"
os.environ['SPARK_HOME']="C:\spark-3.0.0-preview2-bin-hadoop2.7"
sys.path.append("C:\spark-3.0.0-preview2-bin-hadoop2.7\python\pyspark")
from pyspark import SparkConf, SparkContext
conf = SparkConf().setAppName("pyspark text file")
sc = SparkContext(conf=conf)
lines= sc.textFile(r"C:\Users\ravit\Downloads\Applied_Data_Science.txt")
words = lines.flatMap(lambda line: line.split(" "))
words_filter = words.filter(lambda x: len(x)>5)
lwords_filter = words_filter.map(lambda x: x.lower())
wordcount=lwords_filter.map(lambda x:(x,1)) .reduceByKey(lambda x,y: x+y) .map(lambda x: (x[1], x[0])).sortByKey(False)
words=wordcount.top(5)
print("The most used words in the Applied Data Science textbook is:")
for word in words:
    print(word[1],"occured",word[0],"times")