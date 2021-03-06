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

def parse(url):
    try:
        parsed_url_components = url.split('//')
        sublevel_split = parsed_url_components[1].split('/', 1)
        domain = sublevel_split[0].replace("www.", "")
        return domain
    except IndexError:
        print ("URL format error!")

def geturl():
    data_path = r"C:\Users\ravit\AppData\Local\Google\Chrome\User Data\Default"
    files = os.listdir(data_path)
    history_db = os.path.join(data_path, 'history')
    c = sqlite3.connect(history_db)
    cursor = c.cursor()
    select_statement = "SELECT urls.url, urls.visit_count,datetime(visit_time / 1000000 + (strftime('%s', '1601-01-01')), 'unixepoch') FROM urls, visits WHERE urls.id = visits.url "

    cursor.execute(select_statement)

    results = cursor.fetchall()
    countlist = []
    for url, count, date in results:
        result = parse(url)
        if (date > "2020-03-14" and date < "2020-03-21"):
            countlist.append(result)
    return countlist
urllist=geturl()
rdd= sc.parallelize(urllist)
count=rdd.count()
urlcount = rdd.map(lambda x:(x,1)) .reduceByKey(lambda x,y: x+y) .map(lambda x: (x[1], x[0])).sortByKey(False)
print(urlcount.top(5))

