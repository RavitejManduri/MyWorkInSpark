
import os
import sys
import time
import math
from test_helper import Test
import matplotlib.pyplot as plt

baseDir = os.path.join('Downloads')
ratingFilename = os.path.join(baseDir, 'ratings_small.csv')
movieFilename = os.path.join(baseDir, 'movies.csv')
import sys
from pyspark import SparkConf, SparkContext
conf=SparkConf().setMaster("local").setAppName("Movie recommendation")
sc= SparkContext(conf=conf)
numPartition = 2
rawRatings = sc.textFile(ratingFilename).repartition(numPartition)
rawMovies = sc.textFile(movieFilename)
#movieHeader = rawMovies.first()
rawMovies = rawMovies.filter(lambda x: x != movieHeader)

def getRatingTuple(line):
    items = line.replace("\n", "").split(",")
    try:
        return int(items[0]), int(items[1]), float(items[2])
    except ValueError:
        pass

def getMovieTuple(line):
    items = line.replace("\n", "").split(",")
    try:
        return int(items[0]), items[1]
    except ValueError:
        pass

ratingsRDD = rawRatings.map(getRatingTuple).cache()
moviesRDD = rawMovies.map(getMovieTuple).cache()

ratingsCount = ratingsRDD.count()
moviesCount = moviesRDD.count()
