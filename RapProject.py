# coding=utf-8
# Nikita Andreikin

import numpy as np
import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer
from scipy.spatial import distance
from os.path import join
import time
import copy
import math
import random
from sklearn.tree import DecisionTreeClassifier
from sklearn import neural_network
import warnings
from sklearn import metrics
warnings.simplefilter('ignore', RuntimeWarning)
pd.options.mode.chained_assignment = None  # default='warn' Removing Warning

filePath = r'C:\Users\Nikita\Desktop\CS484 Semester Project nandreik\Dataset'   # point to dir of dataset
dataFile = 'songs_dataset.csv'                                                  # name of data set
finalResult = 'result.txt'                                                      # the output of the algorithm's run
track_results = "track_of_runs.txt"                                             # track scores of each run in this file


def read_data():                    # read original song data
    data = pd.read_csv(join(filePath, dataFile))
    return data


def write_data(result, algtype):    # write the result of an algorithm to use for accuracy measure
    subFile = open(join(filePath, finalResult), "w+")
    if algtype == "knn":
        for i in range(len(result)):
            subFile.write(result[i])
            subFile.write("\n")
    if algtype == "tree" or algtype == "nn":
        for i in range(len(result)):
            subFile.write(result[i])
            subFile.write("\n")
    subFile.close()


def save_result(algType, dataType, accuracy, prec, rec, f1, k, wordF, wordL):  # save results of runs in one file
    with open(join(filePath, track_results), "a") as subFile:
        subFile.write("Alg: ")
        subFile.write(str(algType))
        subFile.write("\t\t\tData: ")
        subFile.write(str(dataType))
        subFile.write("\t\t\tAcc: ")
        subFile.write(str(accuracy))
        subFile.write("\t\t\tPrec: ")
        subFile.write(str(prec))
        subFile.write("\t\t\tRec: ")
        subFile.write(str(rec))
        subFile.write("\t\t\tF1: ")
        subFile.write(str(f1))
        if algType == "knn":
            subFile.write("\t")
            subFile.write(" K: " + str(k))
            subFile.write(" WF: " + str(wordF))
            subFile.write(" WL: " + str(wordL))
        else:
            subFile.write("\t")
            subFile.write(" WF: " + str(wordF))
            subFile.write(" WL: " + str(wordL))
        subFile.write("\n")


def convert_to_num(string):         # helper for converting different strings to numbers
    string = string.strip()
    if "." in string:
        res = float(string)
    elif string.isdigit():
        res = int(string)
    else:
        res = string
    return res


def split_data(vectorData, genreData, numSplits):  # split data into numSplits folds
    songSplits = []
    genreSplits = []
    data_size = len(vectorData)
    splitAmt = int(data_size / numSplits)           # get split size
    splitFrom = 0
    splitTo = splitAmt
    for i in range(numSplits):                      # split data into numSplits sections
        if i == numSplits:
            songSplit = vectorData[splitFrom:]
            genreSplit = genreData[splitFrom:]
        else:
            songSplit = vectorData[splitFrom:splitTo]
            genreSplit = genreData[splitFrom:splitTo]
        songSplits.append(songSplit)
        genreSplits.append(genreSplit)
        splitFrom += splitAmt + 1
        splitTo += splitAmt + 1
    return songSplits, genreSplits


def format_data(originalData):                      # format data to be processed
    split = originalData.loc[0:]
    splitLyrics = split["Lyrics"].values
    splitSubGenres = split["Subgenre"].values
    for j in range(len(splitLyrics)):               # format the lyrics/subgenres of each song in the split
        split["Subgenre"].loc[j] = format_string(splitSubGenres[j], False)  # Removed Warning for this above
        split["Lyrics"].loc[j] = format_string(splitLyrics[j], True)
    return split


def format_string(string, isLyric):
    if isLyric:                                         # only remove words between brackets for lyrics
        string = re.sub("[\(\[].*?[\)\]]", "", string)  # added to remove brackets and text in brackets in lyrics [Brackets arent part of lyrics]
    string = re.sub('[\[\]*.,!:?\"\'«»]', '', string)   # strip string for better tokenization
    string = re.sub('[-–——]+', ' ', string)
    string = string.strip().lower()
    string = string.replace('\n', ' ')                  # this was added to replace all \n in lyrics with a space
    return string


def process_data(data, wordFrq, wordLen, dataType):     # process song lyrics and create vectors for algorithms
    bagWords = []                                       # bag of words
    wordCount = []                                      # word count for each word in bag of words
    wordCountVectors = []                               # word counts by song
    wordVectors = []                                    # list of arrays of words in each song
    songVectors = []                                    # list of song vectors that hold word counts for each word in BagWords for corresponding song
    subGenres = []                                      # subgenres of each song from 0 to n in split
    numSongs = 0                                        # track number of songs in this split
    totalWords = 0
    maxSongLyricLength = -1
    stopWords = set(stopwords.words("english"))
    lemma = WordNetLemmatizer()
    stem = PorterStemmer()
    for i in range(len(data)):                          # process each song and its sub-genres in data split
        # if i % 100 == 0:
        #     print "Processed Songs: ", i
        numSongs += 1
        lyrics = word_tokenize(data["Lyrics"].loc[i])   # split lyrics into separate word tokens
        lyrics = [word for word in lyrics if word.isalpha() and word not in stopWords]  # take out numbers/stopwords
        subGenres.append(data["Subgenre"].loc[i])       # append the subgenres for this current song
        songVectorWords = []                            # holds small bag of words for each song
        songVectorCount = []
        for word in lyrics:                             # for each word in lyrics
            word = stem.stem(word)                      # stem word
            word = lemma.lemmatize(word)                # lemmatize word, probably has minimal effect due to slang
            if len(word) >= wordLen:
                if word not in bagWords:                # add word to bag of words
                    bagWords.append(word)
                    wordCount.append(1)
                else:                                   # if already in BoW, inc its count
                    index = bagWords.index(word)
                    wordCount[index] += 1
                if word not in songVectorWords:         # add word to song vector
                    songVectorWords.append(word)
                    songVectorCount.append(1)
                else:                                   # if already in vector, inc its count
                    index = songVectorWords.index(word)
                    songVectorCount[index] += 1
        if len(songVectorCount) > maxSongLyricLength:  # check max song length (NOT SURE IF NEEDED)
            maxSongLyricLength = len(songVectorCount)
        # check word counts in song vectors to be >= wf
        newSongVectCount = []                           # temp vect arrays
        newSongVectWords = []
        for i in range(len(songVectorCount)):
            if songVectorCount[i] >= wordFrq:           # if word count if >= required word frequency, keep word in vector
                newSongVectCount.append(songVectorCount[i])
                newSongVectWords.append(songVectorWords[i])
        wordCountVectors.append(newSongVectCount)
        wordVectors.append(newSongVectWords)

    newWords = []                                       # new bag of words and word count, after removing words under word frequency threshold
    newWordCount = []
    for j in range(len(bagWords)):                      # check word counts in bag words
        if wordCount[j] >= wordFrq:
            newWords.append(bagWords[j])
            newWordCount.append(wordCount[j])
            totalWords += wordCount[j]                  # add to total word count
    bagWords = newWords
    wordCount = newWordCount

    # now that the bag of words and word count is finalized, vectorize the songs
    print "Vectorizing.. "
    for i in range(len(wordVectors)):                   # for each word song vector, convert it to a word count vector of ALL words in bag of words
        # if i % 100 == 0:
            # print "Vectors: ", i
        songVector = np.zeros(len(bagWords))
        wordVector = wordVectors[i]
        wordCountVector = wordCountVectors[i]
        for j in range(len(wordVector)):                # for each word in vector
            word = wordVector[j]                        # get the word from the current song's words
            wordC = wordCountVector[j]                  # get that word's count in the current song
            wordIndBagOfWords = bagWords.index(word)    # get words index in BOW
            songVector[wordIndBagOfWords] = wordC       # update that song's index in the songVector with the current song's word count
        songVectors.append(songVector)                  # add the finished song vector to the list of song vectors

    numSongsWithWord = np.zeros(len(bagWords))          # count of songs that have this word in it
    for vector in wordVectors:                          # for each array of words of each song
        for word in vector:                             # for each word in array
            if word in bagWords:                        # if word in bagwords, update its count for number of songs it is in
                wordInd = bagWords.index(word)
                numSongsWithWord[wordInd] += 1

    tfVectors = []                                      # calc tfidf
    idfVectors = []
    tfidfVectors = []
    if dataType == "tfidf":
        print "Calculating TFIDF Vectors.. "
        count = 0
        for vector in wordVectors:                      # for each word vector
            # if count % 100:
            #     print "TFIDF Vectors: ", count
            count += 1
            tfV = np.zeros(len(bagWords))
            idfV = np.zeros(len(bagWords))
            tfidfV = np.zeros(len(bagWords))
            for i in range(len(vector)):                # for each word in vector
                word = vector[i]
                wordInd = bagWords.index(word)          # calc the tfidf value for each word in that vector
                tf = float(wordCount[wordInd]) / float(totalWords)
                idf = math.log(float(numSongs) / float(numSongsWithWord[wordInd] + 1))
                tfidf = tf * idf
                tfV[wordInd] = tf
                idfV[wordInd] = idf
                tfidfV[wordInd] = tfidf
            tfVectors.append(tfV)                       # add the calculated vectors to the vectors lists
            idfVectors.append(idfV)
            tfidfVectors.append(tfidfV)
    return bagWords, wordCount, subGenres, numSongs, songVectors, maxSongLyricLength, tfidfVectors


def process_genres_to_single(data):         # convert multilabeled subgenres to single labels by artist
    singleDataGenres = [""]*len(data["Subgenre"].values)
    artistList = []
    subGenreCount = [0]*10
    subGenreNames = ["trap", "r&b", "alternative", "drill", "east", "west", "gangsta", "uk", "cloud", "atlanta"]
    print "Calculating Major Genre of Artists..."
    for i in range(len(data)):              # give each artist's song ONE genre label, based on the artist's majority subgenres
        artist = data["Artist"].loc[i]
        if artist not in artistList:        # if artist has not be processed yet, process
            artistList.append(artist)
            artistIndexesInData = []
            for j in range(len(data)):      # for every song by that artist, count the subgenres
                if data["Artist"].loc[j] == artist:
                    artistIndexesInData.append(j)
                    genre = data["Subgenre"].loc[j]
                    if "trap" in genre:     # inc count for each subgenre in the song
                        subGenreCount[0] += 1
                    if "r&b" in genre:
                        subGenreCount[1] += 1
                    if "alternative" in genre:
                        subGenreCount[2] += 1
                    if "drill" in genre:
                        subGenreCount[3] += 1
                    if "east" in genre:
                        subGenreCount[4] += 1
                    if "west" in genre:
                        subGenreCount[5] += 1
                    if "gangsta" in genre:
                        subGenreCount[6] += 1
                    if "uk" in genre:
                        subGenreCount[7] += 1
                    if "cloud" in genre:
                        subGenreCount[8] += 1
                    if "atlanta" in genre:
                        subGenreCount[9] += 1
            # print subGenreCount
            majorityGenreCount = -1
            majorityGenre = ""
            for k in range(len(subGenreCount)):     # find most popular genre by count
                if subGenreCount[k] > majorityGenreCount:
                    majorityGenreCount = subGenreCount[k]
                    majorityGenre = subGenreNames[k]
            subGenreCount = [0] * 10
            # print majorityGenre
            for m in range(len(artistIndexesInData)):   # update all song's genres of that artist to the majority genre
                index = artistIndexesInData[m]
                singleDataGenres[index] = majorityGenre
    return singleDataGenres


def sample_data(songVectors, subgenres):    # total songs in orig data set = 5852
    numSamples = len(subgenres) / 10        # num of samples of each category to take, ~580
    sampleCounts = np.zeros(10)             # count for number of samples of each genre
    vectorsList = []
    subGenresList = []
    notEnoughSamples = True
    while notEnoughSamples:                 # if not enough samples for each genre
        notEnoughSamples = False
        for i in range(len(sampleCounts)):
            if sampleCounts[i] != numSamples:
                notEnoughSamples = True
        for j in range(len(subgenres)):     # add songs until all genres have 100 samples
            genre = str(subgenres[j])
            song = songVectors[j]
            rand = random.uniform(0, 1)     # adding a little randomization to what samples get picked
            randThreshold = .75
            # check genre and sample count before adding sample
            if "trap" in genre and sampleCounts[0] < numSamples:
                if "r&b" not in genre:  # this statement is in each if statement  to limit how often r&b and trap are sampled
                                        # many of the non trap or rnb subgenres have trap or r&n in their subgenre, which creates a skew in the labels
                    if rand >= randThreshold:
                        vectorsList.append(song)
                        subGenresList.append(genre)
                        sampleCounts[0] += 1
            if "r&b" in genre and sampleCounts[1] < numSamples:
                if "trap" not in genre:
                    if rand >= randThreshold:
                        vectorsList.append(song)
                        subGenresList.append(genre)
                        sampleCounts[1] += 1
            if "alternative" in genre and sampleCounts[2] < numSamples:
                if "trap" not in genre and "r&b" not in genre:
                    if rand >= randThreshold:
                        vectorsList.append(song)
                        subGenresList.append(genre)
                        sampleCounts[2] += 1
            if "drill" in genre and sampleCounts[3] < numSamples:
                if "trap" not in genre and "r&b" not in genre:
                    if rand >= randThreshold:
                        vectorsList.append(song)
                        subGenresList.append(genre)
                        sampleCounts[3] += 1
            if "east" in genre and sampleCounts[4] < numSamples:
                if "trap" not in genre and "r&b" not in genre:
                    if rand >= randThreshold:
                        vectorsList.append(song)
                        subGenresList.append(genre)
                        sampleCounts[4] += 1
            if "west" in genre and sampleCounts[5] < numSamples:
                if "trap" not in genre and "r&b" not in genre:
                    if rand >= randThreshold:
                        vectorsList.append(song)
                        subGenresList.append(genre)
                        sampleCounts[5] += 1
            if "gangsta" in genre and sampleCounts[6] < numSamples:
                if "trap" not in genre and "r&b" not in genre:
                    if rand >= randThreshold:
                        vectorsList.append(song)
                        subGenresList.append(genre)
                        sampleCounts[6] += 1
            if "uk" in genre and sampleCounts[7] < numSamples:
                if "trap" not in genre and "r&b" not in genre:
                    if rand >= randThreshold:
                        vectorsList.append(song)
                        subGenresList.append(genre)
                        sampleCounts[7] += 1
            if "cloud" in genre and sampleCounts[8] < numSamples:
                if "trap" not in genre and "r&b" not in genre:
                    if rand >= randThreshold:
                        vectorsList.append(song)
                        subGenresList.append(genre)
                        sampleCounts[8] += 1
            if "atlanta" in genre and sampleCounts[9] < numSamples:
                if "trap" not in genre and "r&b" not in genre:
                    if rand >= randThreshold:
                        vectorsList.append(song)
                        subGenresList.append(genre)
                        sampleCounts[9] += 1
    return vectorsList, subGenresList


def knn(k, trainData, subGenres, testData):
    knnAr = [0] * k                                 # knn list
    knnInd = [0] * k                                # indexes of knns
    knnGenres = [""] * k                            # track sub genres of each Knn Songs
    testResult = [""] * len(testData)               # holds subgenre result for each test vector
    testInd = 0                                     # index for testVectors
    trainInd = 0                                    # index for trainVectors
    resultInd = 0                                   # index for final result array
    subGenreNames = ["trap", "r&b", "alternative", "drill", "east", "west", "gangsta", "uk", "cloud", "atlanta"]
    subGenreCount = [0] * len(subGenreNames)        # list of counts for each of the ten subgenres
    for test in testData:
        for train in trainData:
            testV = np.asarray(test)
            trainV = np.asarray(train)
            # dist = distance.euclidean(test, train) # get distance measure of test and train vectors
            dist = 1 - distance.cosine(testV, trainV)
            if math.isnan(dist):                    # set nan's to 0, nans appear bc some vectors may be all 0s
                dist = 0
            for i in range(len(knnAr)):             # for each k in knn, update knn array w/ new dist
                if knnAr[i] < dist:
                    knnAr[i] = dist                 # update Knn's distance value
                    knnInd[i] = trainInd            # track that song's index in the training data
                    knnGenres[i] = subGenres[trainInd]  # track that song's subgenres
                    break
            trainInd += 1
        for j in range(len(knnAr)):                 # check subgenres from knn array
            if "trap" in knnGenres[j]:              # inc count for each subgenre in the song
                subGenreCount[0] += 1
            if "r&b" in knnGenres[j]:
                subGenreCount[1] += 1
            if "alternative" in knnGenres[j]:
                subGenreCount[2] += 1
            if "drill" in knnGenres[j]:
                subGenreCount[3] += 1
            if "east" in knnGenres[j]:
                subGenreCount[4] += 1
            if "west" in knnGenres[j]:
                subGenreCount[5] += 1
            if "gangsta" in knnGenres[j]:
                subGenreCount[6] += 1
            if "uk" in knnGenres[j]:
                subGenreCount[7] += 1
            if "cloud" in knnGenres[j]:
                subGenreCount[8] += 1
            if "atlanta" in knnGenres[j]:
                subGenreCount[9] += 1
        # print "Genre Counts: ", subGenreCount
        maxCount = -1
        genreLabel = ""
        for u in range(len(subGenreCount)):         # get most popular subgenre count
            if subGenreCount[u] > maxCount:
                maxCount = subGenreCount[u]
        for z in range(len(subGenreCount)):         # classify song as most popular genre out of the k neighbors
            if subGenreCount[z] == maxCount:        # if there are ties in counts with multi. genres, pick the first one encountered in the for loop
                genreLabel += subGenreNames[z]      # *this should work better with higher K's**
                break
            testResult[resultInd] = genreLabel
        subGenreCount = [0] * 10                    # update/reset counts and indexes
        resultInd += 1
        trainInd = 0
        testInd += 1
        for i in range(len(knnAr)):
            knnAr[i] = 0
            knnInd[i] = 0
            knnGenres[i] = ""
        # if testInd % 100 == 0:
        #     print "Number Tested: ", testInd
    return testResult


def get_train_test_splits(wordCountVectors, subGenres):  # get 4 folds of train data and test data
    subGenresTrain1 = copy.copy(subGenres[0])  # train data 1
    for x in subGenres[1]:
        subGenresTrain1.append(x)
    for x in subGenres[2]:
        subGenresTrain1.append(x)
    subGenresTrain2 = copy.copy(subGenres[1])  # train data 2
    for x in subGenres[2]:
        subGenresTrain2.append(x)
    for x in subGenres[3]:
        subGenresTrain2.append(x)
    subGenresTrain3 = copy.copy(subGenres[2])  # train data 3
    for x in subGenres[3]:
        subGenresTrain3.append(x)
    for x in subGenres[0]:
        subGenresTrain3.append(x)
    subGenresTrain4 = copy.copy(subGenres[3])  # train data 4
    for x in subGenres[0]:
        subGenresTrain4.append(x)
    for x in subGenres[1]:
        subGenresTrain4.append(x)

    wordCountTrain1 = copy.copy(wordCountVectors[0])  # train data 1
    for x in wordCountVectors[1]:
        wordCountTrain1.append(x)
    for x in wordCountVectors[2]:
        wordCountTrain1.append(x)
    wordCountTrain2 = copy.copy(wordCountVectors[1])  # train data 2
    for x in wordCountVectors[2]:
        wordCountTrain2.append(x)
    for x in wordCountVectors[3]:
        wordCountTrain2.append(x)
    wordCountTrain3 = copy.copy(wordCountVectors[2])  # train data 3
    for x in wordCountVectors[3]:
        wordCountTrain3.append(x)
    for x in wordCountVectors[0]:
        wordCountTrain3.append(x)
    wordCountTrain4 = copy.copy(wordCountVectors[3])  # train data 4
    for x in wordCountVectors[0]:
        wordCountTrain4.append(x)
    for x in wordCountVectors[1]:
        wordCountTrain4.append(x)

    wordCountTest1 = copy.copy(wordCountVectors[3])  # test data for train corresponding train data 1 - 4
    wordCountTest2 = copy.copy(wordCountVectors[0])  # if using train data 1, use test data 1
    wordCountTest3 = copy.copy(wordCountVectors[1])
    wordCountTest4 = copy.copy(wordCountVectors[2])
    subGenreTest1 = copy.copy(subGenres[3])  # subgenres for test data to evaluate their accuracy
    subGenreTest2 = copy.copy(subGenres[0])
    subGenreTest3 = copy.copy(subGenres[1])
    subGenreTest4 = copy.copy(subGenres[2])

    return (subGenresTrain1, subGenresTrain2, subGenresTrain3, subGenresTrain4,
            wordCountTrain1, wordCountTrain2, wordCountTrain3, wordCountTrain4,
            wordCountTest1, wordCountTest2, wordCountTest3, wordCountTest4,
            subGenreTest1, subGenreTest2, subGenreTest3, subGenreTest4)


def train_tree(X, y):
    tree = DecisionTreeClassifier(criterion="entropy",  # def = "gini"
                                  random_state=484,
                                  max_depth=None,  # def = None
                                  min_samples_split=2,  # def = 2
                                  min_samples_leaf=50,  # def = 1
                                  max_leaf_nodes=None)  # def = None
    tree.fit(X, y)
    return tree


def train_mlp_nn(X, y):
    nn = neural_network.MLPClassifier(solver="sgd",  # def = "adam"
                                      activation="logistic",  # def = "relu"
                                      hidden_layer_sizes=(100,),  # def = (100, )
                                      max_iter=200,  # def = 200
                                      random_state=484)
    nn.fit(X, y)
    return nn


def prepare_data(dataType, wf, wl):
    print "Processing Data.."
    data = read_data()                              # read data
    dataFormatted = format_data(data)               # format data
    bagWords, wordCount, subGenres, numSongs, songVectors, maxSongLyricLength, tfidfVectors = process_data(dataFormatted, wf, wl, dataType)  # process data
    subGenres = process_genres_to_single(data)      # convert subgenres to be single labelled
    if dataType == "unchanged":                     # get train and test splits using only word count
        print "Data Unchanged.."
        songSplits, genreSplits = split_data(songVectors, subGenres, 4)
        (subGenresTrain1, subGenresTrain2, subGenresTrain3, subGenresTrain4,
         wordCountTrain1, wordCountTrain2, wordCountTrain3, wordCountTrain4,
         wordCountTest1, wordCountTest2, wordCountTest3, wordCountTest4,
         subGenreTest1, subGenreTest2, subGenreTest3, subGenreTest4) = get_train_test_splits(songSplits, genreSplits)
    if dataType == "tfidf":                         # get train and test split using tfidf values instead of word count
        print "Calculating TFIDF.."
        songSplits, genreSplits = split_data(tfidfVectors, subGenres, 4) # split data on tfidf vectors
        (subGenresTrain1, subGenresTrain2, subGenresTrain3, subGenresTrain4,
         wordCountTrain1, wordCountTrain2, wordCountTrain3, wordCountTrain4,
         wordCountTest1, wordCountTest2, wordCountTest3, wordCountTest4,
         subGenreTest1, subGenreTest2, subGenreTest3, subGenreTest4) = get_train_test_splits(songSplits, genreSplits)
    if dataType == "sampled":                       # sample genres so that popular genres don't overwhelm training
        print "Sampling Data.."
        newVectorsList, newSubGenres = sample_data(songVectors, subGenres)      # sample data
        songSplits, genreSplits = split_data(newVectorsList, newSubGenres, 4)   # split data
        (subGenresTrain1, subGenresTrain2, subGenresTrain3, subGenresTrain4,
         wordCountTrain1, wordCountTrain2, wordCountTrain3, wordCountTrain4,
         wordCountTest1, wordCountTest2, wordCountTest3, wordCountTest4,
         subGenreTest1, subGenreTest2, subGenreTest3, subGenreTest4) = get_train_test_splits(songSplits, genreSplits)
    return (subGenresTrain1, subGenresTrain2, subGenresTrain3, subGenresTrain4,
            wordCountTrain1, wordCountTrain2, wordCountTrain3, wordCountTrain4,
            wordCountTest1, wordCountTest2, wordCountTest3, wordCountTest4,
            subGenreTest1, subGenreTest2, subGenreTest3, subGenreTest4)


def class_report(actualSubgenres, predictedSubgenres):  # create confusion matrix and evaluate it with Sklearn
    labels = ["trap", "r&b", "alternative", "drill", "east coast", "west coast", "gangsta rap", "uk", "cloud rap", "atlanta"]
    report = metrics.classification_report(actualSubgenres, predictedSubgenres, labels=labels)
    matrix = metrics.confusion_matrix(actualSubgenres, predictedSubgenres, labels=labels)
    accuracy = metrics.accuracy_score(actualSubgenres, predictedSubgenres)
    precRecF1Weight = metrics.precision_recall_fscore_support(actualSubgenres, predictedSubgenres, labels=labels, average="weighted")
    prec = precRecF1Weight[0]
    rec = precRecF1Weight[1]
    f1 = precRecF1Weight[2]
    print "Weighted: ", precRecF1Weight
    print matrix
    print report
    print "Accuracy: ", accuracy
    print "Precision: ", precRecF1Weight[0]
    print "Recall: ", precRecF1Weight[1]
    print "F1-Score: ", precRecF1Weight[2]
    return report, accuracy, prec, rec, f1


def main():
    print "START"
    start = time.clock()
    alg = "knn"                                 # pick what algorithm to run: "knn", "tree", "nn"
    k = 3                                       # knn number
    wf = 2                                      # min word freq limit
    wl = 3                                      # min word length limit
    dataType = "unchanged"                      # pick type of data to use: "unchanged", "tfidf", "sampled"

    # process and split the data
    (subGenresTrain1, subGenresTrain2, subGenresTrain3, subGenresTrain4,
     wordCountTrain1, wordCountTrain2, wordCountTrain3, wordCountTrain4,
     wordCountTest1, wordCountTest2, wordCountTest3, wordCountTest4,
     subGenreTest1, subGenreTest2, subGenreTest3, subGenreTest4) = prepare_data(dataType, wf, wl)

    # choose which data folds to train and test with
    wordCountTrain = wordCountTrain1
    subGenresTrain = subGenresTrain1
    wordCountTest = wordCountTest1
    subGenreTest = subGenreTest1

    print "Running ", alg, " Algorithm"
    if alg == "knn":                            # run knn
        result = knn(k, wordCountTrain, subGenresTrain, wordCountTest)
        write_data(result, "knn")
    if alg == "tree":                           # run dec tree
        tree = train_tree(wordCountTrain, subGenresTrain)
        result = tree.predict(wordCountTest)
        write_data(result, "tree")
    if alg == "nn":                             # run NN
        nn = train_mlp_nn(wordCountTrain, subGenresTrain)
        result = nn.predict(wordCountTest)
        write_data(result, "nn")
    report, accuracy, prec, rec, f1 = class_report(subGenreTest, result)    # get the report of the confusion matrix
    save_result(alg, dataType, accuracy, prec, rec, f1, k, wf, wl)          # save result
    print "DONE"
    done = (time.clock() - start)
    print done/60, " Minutes"


if __name__ == "__main__":
    main()
