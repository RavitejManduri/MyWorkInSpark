import os
import sys
import time
import math

import matplotlib.pyplot as plt
os.environ["SPARK_HOME"] = "/usr/local/spark"
os.environ['SPARK_HOME']="C:\spark-3.0.0-preview2-bin-hadoop2.7"
# Append pyspark  to Python Path
sys.path.append("C:\spark-3.0.0-preview2-bin-hadoop2.7\python\pyspark")
from pyspark import SparkConf, SparkContext
conf=SparkConf().setMaster("local").setAppName("Final Project")
sc = SparkContext(conf=conf)
from pyspark.sql import SQLContext
baseDir = os.path.join(r'C:\Users\ravit\Downloads\ml-latest-small')
ratingFilename = os.path.join(baseDir, 'ratings.csv')
movieFilename = os.path.join(baseDir, 'movies.csv')

numPartition = 2
rawRatings = sc.textFile(ratingFilename)
rawMovies = sc.textFile(movieFilename,10)
movieHeader = rawMovies.first()
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

def calcUserMeanRating(userRatingGroup):
    """ Calculate the average rating of a user
    """
    userID = userRatingGroup[0]
    ratingSum = 0.0
    ratingCnt = len(userRatingGroup[1])
    if ratingCnt == 0:
        return (userID, 0.0)
    for item in userRatingGroup[1]:
        ratingSum += item[1]
    return (userID, 1.0 * ratingSum / ratingCnt)


def broadcastUserRatingAvg(sContext, uRRDDTrain):
    """ Broadcast the user average rating RDD
    """
    userRatingAvgList = uRRDDTrain.map(lambda x: calcUserMeanRating(x)).collect()
    userRatingAvgDict = {}
    for (user, avgscore) in userRatingAvgList:
        userRatingAvgDict[user] = avgscore
    uRatingAvgBC = sContext.broadcast(userRatingAvgDict)# broadcast
    return uRatingAvgBC


def broadcastUMHist(sContext, uRRDDTrain):
    """ Broadcast user movie history dict
    """
    userMovieHistList = uRRDDTrain.map(lambda x: constructUserMovieHist(x)).collect()
    userMovieHistDict = {}
    for (user, mrlistTuple) in userMovieHistList:
        userMovieHistDict[user] = mrlistTuple
    uMHistBC = sContext.broadcast(userMovieHistDict)# broadcast
    return uMHistBC

def constructUserMovieHist(userRatingGroup):
    """ Construct the rating list of a user
    Returns:
        (user, ([movie], [rating]))
    """
    userID = userRatingGroup[0]
    movieList = [item[0] for item in userRatingGroup[1]]
    ratingList = [item[1] for item in userRatingGroup[1]]
    return (userID, (movieList, ratingList))

def constructCommonRating(tup1, tup2):
    """
    Args:
        tup1 and tup2 are of the form (user, [(movie, rating)])
    Returns:
        ((user1, user2), [(rating1, rating2)])
    """
    user1, user2 = tup1[0], tup2[0]
    mrlist1 = sorted(tup1[1])
    mrlist2 = sorted(tup2[1])
    ratepair = []
    index1, index2 = 0, 0
    while index1 < len(mrlist1) and index2 < len(mrlist2):
        if mrlist1[index1][0] < mrlist2[index2][0]:
            index1 += 1
        elif mrlist1[index1][0] == mrlist2[index2][0]:
            ratepair.append((mrlist1[index1][1], mrlist2[index2][1]))
            index1 += 1
            index2 += 1
        else:
            index2 += 1
    return ((user1, user2), ratepair)

def calcCosineSimilarity(tup):
    """ Compute cosine similarity
    Args:
        tup: ((user1, user2), [(rating1, rating2)])
    Returns:
        ((user1, user2), (similarity, number of common ratings))
    """
    dotproduct = 0.0
    sqsum1, sqsum2, cnt = 0.0, 0.0, 0
    for rpair in tup[1]:
        dotproduct += rpair[0] * rpair[1]
        sqsum1 += (rpair[0]) ** 2
        sqsum2 += (rpair[1]) ** 2
        cnt += 1
    denominator = math.sqrt(sqsum1) * math.sqrt(sqsum2)
    similarity = (dotproduct / denominator) if denominator else 0.0
    return (tup[0], (similarity, cnt))


def keyOnUser(record):
    """
    Args:
        record: ((user1, user2), (similarity, #movies both rated))
    Returns:
        [(user1, (user2, similarity, #movies both rated)), (user2, (user1, similarity, #movies both rated))]
    """
    return [(record[0][0], (record[0][1], record[1][0], record[1][1])),
            (record[0][1], (record[0][0], record[1][0], record[1][1]))]

def getTopKSimilarUser(user, records, numK = 200):
    """
    Args:
        user: id of a user
        records: [(user_sim, similarity, number of common ratings)]
        numK: number of similar users we want to keep track of
    Returns:
        (user, [(user_sim, similarity, number of common ratings)])
    """
    llist = sorted(records, key=lambda x: x[1], reverse=True)
    llist = [x for x in llist if x[2] > 9]# filter out those whose cnt is small
    return (user, (llist[:numK]))

def broadcastUNeighborDict(sContext, uNeighborRDD):
    """ Broadcast user neighbors dict
    """
    userNeighborList = uNeighborRDD.collect()
    userNeighborDict = {}
    for user, simrecords in userNeighborList:
        userNeighborDict[user] = simrecords
    uNeighborBC = sContext.broadcast(userNeighborDict)# broadcast
    return uNeighborBC

from collections import defaultdict

def recommendUB(user, neighbors, usermovHistDict, topK = 200, nRec = 30):
    """ User based recommendation
    maintain two dicts, one for similarity sum, one for weighted rating sum
    for every neighbor of a user, get his rated items which hasn't been rated by current user
    then for each movie, sum the weighted rating in the whole neighborhood
    and sum the similarity of users who rated the movie
    iterate and sort
    Args:
        user: id of a user asking for recommendation
        neighbors: [(user_sim, similarity, number of common ratings)]
        usermovHistDict: (user, ([movie], [rating]))
        topK: the number of neighbors to use
        nRec: the number of recommendation
    """
    simSumDict = defaultdict(float)# similarity sum
    weightedSumDict = defaultdict(float)
    # weighted rating sum
    movIDUserRated = usermovHistDict.get(user, [])
    for (neighbor, simScore, numCommonRating) in neighbors[:topK]:
        mrlistpair = usermovHistDict.get(neighbor)
        if mrlistpair:
            for index in range(0, len(mrlistpair[0])):
                movID = mrlistpair[0][index]
                simSumDict[movID] += simScore
                weightedSumDict[movID] += simScore * mrlistpair[1][index]# sim * rating
    candidates = [(mID, 1.0 * weightedSumDict[mID] / simSumDict[mID]) for (mID) in weightedSumDict]
    candidates.sort(key=lambda x: x[1], reverse=True)
    return (user, candidates[:nRec])

def broadcastMovNameDict(sContext, movRDD):
    movieNameList = movRDD.collect()
    movieNameDict = {}
    for (movID, movName) in movieNameList:
        movieNameDict[movID] = movName
    mNameDictBC = sc.broadcast(movieNameDict)
    return mNameDictBC

def genMovRecName(user, records, movNameDict):
    nlist = []
    for record in records:
        nlist.append(movNameDict[record[0]])#userRecomMovNamesRDD
    return (user, nlist)


ratingsRDD = rawRatings.map(lambda l: getRatingTuple(l)).cache()
moviesRDD = rawMovies.map(lambda l: getMovieTuple(l)).cache()

userRatingRDD = ratingsRDD.map(lambda x: (x[0], (x[1],x[2]))).groupByKey()
userRatingAvgBC = broadcastUserRatingAvg(sc, userRatingRDD)
userMovieHistBC = broadcastUMHist(sc, userRatingRDD)
cartesianRDD = userRatingRDD.cartesian(userRatingRDD)


userPairRawRDD = cartesianRDD.filter(lambda x1: x1[0][0]<x1[1][0])
userPairRDD = userPairRawRDD.map(
    lambda x1: constructCommonRating(x1[0], x1[1]))
print(userPairRDD.take(3))
userSimilarityRDD = userPairRDD.map(lambda x: calcCosineSimilarity(x))
print(userSimilarityRDD.take(5))
userSimGroupRDD = userSimilarityRDD.flatMap(lambda x: keyOnUser(x)).groupByKey()
    # userNeighborRDD: (user, [(user_sim, similarity, number of common ratings)])
userNeighborRDD = userSimGroupRDD.map(lambda x1: getTopKSimilarUser(x1[0], x1[1], 200))

userNeighborBC = broadcastUNeighborDict(sc, userNeighborRDD)
userRecomMovIDsRDD = userNeighborRDD.map(lambda x1: recommendUB(x1[0], x1[1], userMovieHistBC.value))
movieNameDictBC = broadcastMovNameDict(sc, moviesRDD)
userRecomMovNamesRDD = userRecomMovIDsRDD.map(lambda x1: genMovRecName(x1[0], x1[1], movieNameDictBC.value))
#print(userRecomMovNamesRDD .take(5))
print ('Recommend movies using user-based method for user 2: \n')
print (userRecomMovNamesRDD.filter(lambda x1: x1[0] == 2).collect())


