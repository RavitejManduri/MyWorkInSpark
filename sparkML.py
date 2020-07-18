import os
import sys
import sqlite3
os.environ["SPARK_HOME"] = "/usr/local/spark"
os.environ['SPARK_HOME']="C:\spark-3.0.0-preview2-bin-hadoop2.7"
sys.path.append("C:\spark-3.0.0-preview2-bin-hadoop2.7\python\pyspark")
from pyspark import SparkConf, SparkContext
conf = SparkConf().setAppName("pyspark text file")
sc = SparkContext(conf=conf)
from pyspark.sql import SQLContext
from pyspark.sql.types import *
import pandas as pd
import datetime
import random
df = pd.DataFrame(columns=['Flight No','Date','Time','Actual Date','Origin','Destination'])

from random import randrange
import datetime
def random_date(start,l):
   current = start
   while l >= 0:
    current = current + datetime.timedelta(minutes=randrange(10))
    yield current
    l-=1

flightno=[]
for i in range(10):
    s=""
    s="UM-"+str(random.randint(111,999))
    flightno.append(str(s))

airports=["BHM","DHN","HSV","MGM","MRI","BWI","LAX","JFK","MIA","PHL","PSG","KSM","TUS","YUM","OAK"]
#random.sample(airport, 2)
station={}

for i in range(101):
    fno=random.choice(flightno)
    df.at[i,"Flight No"]=fno
    start_date = datetime.date(2020, 1, 1)
    end_date = datetime.date(2020, 2, 1)
    time_between_dates = end_date - start_date
    days_between_dates = time_between_dates.days
    random_number_of_days = random.randrange(days_between_dates)
    random_date = start_date + datetime.timedelta(days=random_number_of_days)
    #print(random_date.strftime("%d/%m/%y"))
    df.at[i,"Date"]=random_date.strftime("%d/%m/%y")
    startDate = datetime.datetime(2013, 9, 20,random.randint(0,23),00)
    current = startDate + datetime.timedelta(minutes=randrange(30))
    df.at[i,"Time"]=current.strftime("%H:%M")
    delay=datetime.timedelta(minutes=randrange(45))
    df.at[i,"Actual Time"]=(current+delay).strftime("%H:%M")
    df.at[i,"Actual Date"]=(random_date+datetime.timedelta(days=randrange(1))).strftime("%d/%m/%y")
    if fno not in station.keys():
        station[fno]=random.sample(airports, 2)
    df.at[i,"Origin"]=station[fno][0]
    df.at[i,"Destination"]=station[fno][1]
    #df.at[i,"Delay"]=(current-current+delay).strftime("%H:%M")
#train, test = data_2.randomSplit([0.7, 0.3])
print(df)
sqlContext = SQLContext(sc)
data=sqlContext.createDataFrame(df)

feature_columns = data.columns[:-1]
from pyspark.ml.feature import VectorAssembler
assembler = VectorAssembler(inputCols=feature_columns,outputCol="features")
data_2 = assembler.transform(data)

train, test = data.randomSplit([0.7, 0.3])


from pyspark.ml.regression import LinearRegression
algo = LinearRegression(featuresCol="features", labelCol="medv")
# train the model
model = algo.fit(train)
evaluation_summary = model.evaluate(test)
evaluation_summary.meanAbsoluteError
evaluation_summary.rootMeanSquaredError
evaluation_summary.r2
# predicting values
predictions = model.transform(test)
predictions.select(predictions.columns[13:]).show()


