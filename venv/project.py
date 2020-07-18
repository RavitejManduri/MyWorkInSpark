class TestFailure(Exception):
  pass
class PrivateTestFailure(Exception):
  pass

class Test(object):
  passed = 0
  numTests = 0
  failFast = False
  private = False

  @classmethod
  def setFailFast(cls):
    cls.failFast = True

  @classmethod
  def setPrivateMode(cls):
    cls.private = True

  @classmethod
  def assertTrue(cls, result, msg=""):
    cls.numTests += 1
    if result == True:
      cls.passed += 1
      print ("1 test passed.")
    else:
      print ("1 test failed. " + msg)
      if cls.failFast:
        if cls.private:
          raise PrivateTestFailure(msg)
        else:
          raise TestFailure(msg)

  @classmethod
  def assertEquals(cls, var, val, msg=""):
    cls.assertTrue(var == val, msg)

  @classmethod
  def assertEqualsHashed(cls, var, hashed_val, msg=""):
    cls.assertEquals(cls._hash(var), hashed_val, msg)

  @classmethod
  def printStats(cls):
    print ("{0} / {1} test(s) passed.".format(cls.passed, cls.numTests))

  @classmethod
  def _hash(cls, x):
    return hashlib.sha1(str(x)).hexdigest()





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

ratingsRDD = rawRatings.map(lambda l: getRatingTuple(l)).cache()
moviesRDD = rawMovies.map(lambda l: getMovieTuple(l)).cache()

ratingsCount = ratingsRDD.count()
moviesCount = moviesRDD.count()

print ('There are %s ratings and %s movies in the small dataset' , (ratingsCount, moviesCount))
# There are 141191 ratings and 27278 movies in the small dataset
print ('Ratings: %s' % ratingsRDD.take(3))
print ('Movies: %s' % moviesRDD.take(3))

def get_rating_distribution(ratingRDD):
    return ratingRDD.map(lambda x: (x[2], 1)).countByKey().items()

ratingscoreDict = {x[0]:x[1] for x in get_rating_distribution(ratingsRDD)}
# print ratingscoreDict
plt.figure()
plt.bar(ratingscoreDict.keys(), ratingscoreDict.values(), 0.5)
plt.title('Rating score distribution')
# compute the average rating score of the whole dataset
avgrating = sum([k * ratingscoreDict[k] for k in ratingscoreDict]) / sum(ratingscoreDict.values())
print ("average rating of the whole dataset: %.2f" % avgrating)

# movie - number of ratings distribution
def getMovieRatingDistribution(ratingRDD):
    tempdict = ratingRDD.map(lambda x: (x[1], 1)).countByKey()
    return sc.parallelize(tempdict.values()).histogram(20)

movieratedDict = getMovieRatingDistribution(ratingsRDD)
# print movieratedDict


def getUserRatingDistribution(ratingRDD):
    tempdict = ratingRDD.map(lambda x: (x[0], 1)).countByKey()
    return sc.parallelize(tempdict.values()).histogram(20)

userratedDict = getUserRatingDistribution(ratingsRDD)

def getRatingDistributionOfAMovie(ratingRDD, movieID):
    """ Get the rating distribution of a specific movie
    Args:
        ratingRDD: a RDD containing tuples of (UserID, MovieID, Rating)
        movieID: the ID of a specific movie
    Returns:
        [(rating score, number of this rating score)]
    """
    return ratingRDD.filter(lambda x: x[1] == movieID).map(lambda x: (x[2], 1)).countByKey()

def getRatingDistributionOfAUser(ratingRDD, userID):
    """ Get the rating distribution of a specific user
    Args:
        ratingRDD: a RDD containing tuples of (UserID, MovieID, Rating)
        userID: the ID of a specific user
    Returns:
        [(rating score, number of this rating score)]
    """
    return ratingRDD.filter(lambda x: x[0] == userID).map(lambda x: (x[2], 1)).countByKey()

print( getRatingDistributionOfAMovie(ratingsRDD, 587))
print (getRatingDistributionOfAUser(ratingsRDD, 1))

# set seeds
seeds = range(10)


def computeErrors(predictedRDD, actualRDD):
    """ Compute the RMSE and MAE between predicted and actual RDD
    Args:
        predictedRDD: predicted ratings for each movie and each user where each entry is in the form
                      (UserID, MovieID, Rating)
        actualRDD: actual ratings where each entry is in the form (UserID, MovieID, Rating)
    Returns:
        (RMSE, MAE)
        RMSE (float): computed RMSE value
        MAE (float): computed MAE value
    """
    # Transform predictedRDD into the tuples of the form ((UserID, MovieID), Rating)
    predictedReformattedRDD = predictedRDD.map(lambda rec: ((rec[0], rec[1]), rec[2]))
    # Transform actualRDD into the tuples of the form ((UserID, MovieID), Rating)
    actualReformattedRDD = actualRDD.map(lambda rec: ((rec[0], rec[1]), rec[2]))
    # Join the two RDD
    joinedRDD = predictedReformattedRDD.join(actualReformattedRDD)
    # Errors
    squaredErrorsRDD = joinedRDD.map(lambda x: (x[1][0] - x[1][1])*(x[1][0] - x[1][1]))
    absErrorsRDD = joinedRDD.map(lambda x: abs(x[1][0] - x[1][1]))
    # Compute the total error
    totalSquareError = squaredErrorsRDD.reduce(lambda v1, v2: v1 + v2)
    totalAbsoluteError = absErrorsRDD.reduce(lambda v1, v2: v1 + v2)
    # Count the number of entries for which you computed the total squared error
    numRatings = squaredErrorsRDD.count()
    return (math.sqrt(float(totalSquareError) / numRatings), float(totalAbsoluteError) / numRatings)


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

def predictUsingAvg(tup, avgDict):
    """ Predict using user's average rating
    """
    user, movie = tup[0], tup[1]
    avgrate = avgDict.get(user, 0.0)
    return (user, movie, avgrate)

baselineErrors = [0] * len(seeds)
err = 0
for seed in seeds:
    trainingRDD, validationRDD, testRDD = ratingsRDD.randomSplit([6, 2, 2], seed=seed)
    trainingRDD = trainingRDD.union(validationRDD)
    # (user, [(movie, rating)])
    userRatingRDDTrain = trainingRDD.map(lambda x: (x[0], (x[1], x[2]))).groupByKey()

    userRatingAvgBC = broadcastUserRatingAvg(sc, userRatingRDDTrain)
    # print 'show some values in userRatingAvgBC: %s' % userRatingAvgBC.value.get(1, 0)
    testForPredictingRDD = testRDD.map(lambda x: (x[0], x[1]))
    predictedAvgRDD = testForPredictingRDD.map(
            lambda x: predictUsingAvg(x, userRatingAvgBC.value))
    baselineErrors[err] = computeErrors(predictedAvgRDD, testRDD)
    err += 1
    # print 'predictedAvgRDD take 3: %s' % predictedAvgRDD.take(3)

blRMSEs, blMAEs = [x[0] for x in baselineErrors], [x[1] for x in baselineErrors]
#print ("Baseline Approach -- Average RMSE on test set: %f" % (sum(blRMSEs)/float(len(blRMSEs))))
#print ("Baseline Approach -- Average MAE on test set: %f" % (sum(blMAEs)/float(len(blMAEs))))


def getCountsAndAverages(IDandRatingsTuple):
    """ Calculate average rating
    Args:
        IDandRatingsTuple: a single tuple of (MovieID, (Rating1, Rating2, Rating3, ...))
    Returns:
        tuple: a tuple of (MovieID, (number of ratings, averageRating))
    """
    rateCnt = len(IDandRatingsTuple[1])
    return (IDandRatingsTuple[0], (rateCnt, float(sum(IDandRatingsTuple[1]))/rateCnt))


movIDRatingsRDD = ratingsRDD.map(lambda x: (x[1], x[2])).groupByKey()
movAvgRatingCntRDD = movIDRatingsRDD.map(getCountsAndAverages)
movNameAvgRatingCntRDD = (moviesRDD
                              .join(movAvgRatingCntRDD).map(lambda x: (x[1][1][1], x[1][0], x[1][1][0])))
movLimitedAndSortedRDD = movNameAvgRatingCntRDD.filter(lambda x: x[2] > 100).sortBy(lambda x: x[0], False)

print ('General Approach -- Recommend movies with highest average rating and more than 100 ratings\n')
print ('\n'.join(map(str, movLimitedAndSortedRDD.take(30))))

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

Test.assertEquals(constructCommonRating(
        (1, [(9, 3.0), (2, 4.0), (1, 3.0), (6, 2.0), (7, 3.5), (5, 2.5)]),
        (2, [(7, 3.0), (3, 4.0), (6, 3.0), (5, 2.0), (2, 3.0), (8, 4.0)])),
                  ((1, 2), [(4.0, 3.0), (2.5, 2.0), (2.0, 3.0), (3.5, 3.0)]),
                            'incorrect constructCommonRating()')
print(constructCommonRating(
        (1, [(9, 3.0), (2, 4.0), (1, 3.0), (6, 2.0), (7, 3.5), (5, 2.5)]),
        (2, [(7, 3.0), (3, 4.0), (6, 3.0), (5, 2.0), (2, 3.0), (8, 4.0)])))

def makeUserPair(record):
    """
    Args:
        record: (movie, [(user, rating)])
    Returns:
        [((user1, user2), (rating1, rating2))]
    """
    ll = sorted(record[1], key=lambda x: x[0])
    length = len(ll)
    pairs = []
    for i in range(0, length-1):
        for j in range(i+1, length):
            pairs.append(((ll[i][0], ll[j][0]), (ll[i][1], ll[j][1])))
    return pairs

Test.assertEquals(makeUserPair((1, [(4, 4), (1, 1), (2, 2), (3, 3)])),
                  [((1, 2), (1, 2)), ((1, 3), (1, 3)), ((1, 4), (1, 4)),
                   ((2, 3), (2, 3)), ((2, 4), (2, 4)), ((3, 4), (3, 4))],
                            'incorrect makeUserPair()')

def constructUserMovieHist(userRatingGroup):
    """ Construct the rating list of a user
    Returns:
        (user, ([movie], [rating]))
    """
    userID = userRatingGroup[0]
    movieList = [item[0] for item in userRatingGroup[1]]
    ratingList = [item[1] for item in userRatingGroup[1]]
    return (userID, (movieList, ratingList))


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







