#!/usr/bin/env python

#  Original Author: Angela Chapman
#  Date: 8/6/2014
#  Modified by Haixia Liu 
#  This file contains code to accompany the Kaggle tutorial
#  "Deep learning goes to the movies".  The code in this file
#  is for Parts 2 and 3 of the tutorial, which cover how to
#  train a model using Word2Vec.
#
# *************************************** #


# ****** Read the two training sets and the test set
#
import pandas as pd
import os
from nltk.corpus import stopwords
import nltk.data
import logging
import numpy as np  # Make sure that numpy is imported
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier

from KaggleWord2VecUtility import KaggleWord2VecUtility
import gensim
import csv

# ****** Define functions to create average word vectors
#

def makeFeatureVec(words, model, num_features):
    # Function to average all of the word vectors in a given
    # paragraph
    #
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,),dtype="float32")
    #
    nwords = 0.
    #
    # Index2word is a list that contains the names of the words in
    # the model's vocabulary. Convert it to a set, for speed
    index2word_set = set(model.index2word)
    #
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
    #
    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec,nwords)
    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
    # Given a set of reviews (each one a list of words), calculate
    # the average feature vector for each one and return a 2D numpy array
    #
    # Initialize a counter
    counter = 0.
    #
    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    #
    # Loop through the reviews
    for review in reviews:
       #
       # Print a status message every 1000th review
       if counter%1000. == 0.:
           print("Review %d of %d" % (counter, len(reviews)))
       #
       # Call the function (defined above) that makes average feature vectors
       reviewFeatureVecs[counter] = makeFeatureVec(review, model, \
           num_features)
       #
       # Increment the counter
       counter = counter + 1.
    return reviewFeatureVecs


def getCleanReviews(reviews):
    clean_reviews = []
    for review in reviews["review"]:
        clean_reviews.append( KaggleWord2VecUtility.review_to_wordlist( review, remove_stopwords=True ))
    return clean_reviews



if __name__ == '__main__':

    # Read data from files
    # train = pd.read_csv( os.path.join(os.path.dirname(__file__), 'data', 'labeledTrainData.tsv'), header=0, delimiter="\t", quoting=3 )
    # train = pd.read_csv( os.path.join(os.path.dirname(__file__), 'data', 'trainACL.csv'), header=0, delimiter="\t", quoting=3 )

# df = pd.read_csv('/path/x.csv', skipinitialspace=True)
# x, y = df['text'], df['label']
    # test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'testData.tsv'), header=0, delimiter="\t", quoting=3 )
    # test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'testACL1000.csv'), header=0, delimiter="\t", quoting=3 )
    # aclALL = pd.read_csv( os.path.join(os.path.dirname(__file__), 'data', 'sentences.csv'), header=0, quoting=3 )
    # aclALL = pd.read_csv( os.path.join(os.path.dirname(__file__), 'data', 'sorted1971-2011.csv'), header=0, quoting=3 )
    aclALL = pd.read_csv( os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/fortrainmodel/', 'ACLPart1.csv'), header=0,delimiter=",", quoting=3 )


    # aclAnnotated = pd.read_csv( os.path.join(os.path.dirname(__file__), 'data', 'ACLAnnotatedTrain.csv'), header=0, quoting=3 )
    # unlabeled_train = pd.read_csv( os.path.join(os.path.dirname(__file__), 'data', "unlabeledTrainData.tsv"), header=0,  delimiter="\t", quoting=3 )

    #
    # unlabeled_train_cff = pd.read_csv( os.path.join(os.path.dirname(__file__), 'data', "aclnocommasepnew.ods"), header=0,  delimiter="\t", quoting=3, skipinitialspace=True, engine='python', dialect=csv.excel_tab)

    # unlabeled_train_cff0 = csv.reader(x.replace('\0', '') for x in "data/aclnocommasepnew.ods")
    # unlabeled_train_cff = pd.read_csv( unlabeled_train_cff0, header=0, delimiter="\t", quoting=3 )

    # unlabeled_train_cff = csv.reader(open("data/mynew.ods", 'rU'), dialect=csv.excel_tab)
    # fullfile=[]
    # with open('data/aclreview.txt') as fp:
    #     for line in fp:
    #         fullfile.append(line)
    #
    # with open('data/acltrainreview.txt') as fp:
    #     for line in fp:
    #         fullfile.append(line)
    # Verify the number of reviews that were read (100,000 in total)
    # print("Read %d labeled train reviews, %d labeled test reviews, " \
    #  "and %d unlabeled reviews\n" % (train["review"].size,
    #  test["review"].size, unlabeled_train["review"].size ))



    # Load the punkt tokenizer
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')



    # ****** Split the labeled and unlabeled training sets into clean sentences
    #
    sentences = []  # Initialize an empty list of sentences
        # print("Parsing sentences from aclALL")
    for review in aclALL["review"]:
        if str(review).strip() != '' and str(review).strip() != ' ':
            sentences += KaggleWord2VecUtility.review_to_sentences(review, tokenizer)

                    # print("Parsing sentences from aclAnnotated")
    # for review in aclAnnotated["review"]:
    #     if str(review).strip() != '' and str(review).strip() != ' ':
    #         sentences += KaggleWord2VecUtility.review_to_sentences(review, tokenizer)

    # print("Parsing sentences from training set")
    # for review in train["review"]:
    #     if str(review).strip() != '' and str(review).strip() != ' ':
    #         sentences += KaggleWord2VecUtility.review_to_sentences(review, tokenizer)
#seems useful, but if solve .csv file:remove deliminator, then, fine
    # print("Parsing sentences from unlabeled set")
    # # for review in unlabeled_train_cff["review"]:
    # for review in fullfile:
    #     if str(review).strip() != '' and str(review).strip() != ' ':
    #         sentences += KaggleWord2VecUtility.review_to_sentences(review, tokenizer)
#seems useful, but if solve .csv file:remove deliminator, then, fine
    # ****** Set parameters and train the word2vec model
    #
    # Import the built-in logging module and configure it so that Word2Vec
    # creates nice output messages
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
        level=logging.INFO)

    # Set values for various parameters
    num_features = 100    # Word vector dimensionality
    min_word_count = 40   # Minimum word count
    num_workers = 4       # Number of threads to run in parallel
    context = 10          # Context window size
    downsampling = 1e-3   # Downsample setting for frequent words

    # #=====================to compare with sen2vec=================
    # # def __init__(self, sentences=None, size=100, alpha=0.025, window=5, min_count=5,
    # # sample=0, seed=1, workers=1, min_alpha=0.0001, sg=1, hs=1, negative=0, cbow_mean=0):
    # num_features = 100    # Word vector dimensionality
    # min_word_count = 5   # Minimum word count
    # num_workers = 1       # Number of threads to run in parallel
    # context = 5          # Context window size
    # downsampling = 0  # Downsample setting for frequent words
    #=========================================================================

    # Initialize and train the model (this will take some time)

    print("Training Word2Vec model...for phrase-haixia modify")

    # model = Word2Vec(sentences, workers=num_workers, \
    #             size=num_features, min_count = min_word_count, \
    #             window = context, sample = downsampling, seed=1)
    #
    bigramtransformer = gensim.models.Phrases(sentences)
    model = Word2Vec(bigramtransformer[sentences], workers=num_workers, vector_size=num_features, min_count = min_word_count, window = context, sample = downsampling, seed=1)


    # If you don't plan to train the model any further, calling
    # init_sims will make the model much more memory-efficient.
    model.init_sims(replace=True)

    # It can be helpful to create a meaningful model name and
    # save the model for later use. You can load it later using Word2Vec.load()
    # model_name = "100features_40minwords_10context_learn2learn_1971-2011_word2vec"
    model_name = "100features_40minwords_10context_arg_word2vec"
    model.save(model_name)

    # model.doesnt_match("canopy development population structure seasonal ecophysiology soil hydrology determine".split())
    # model.doesnt_match("france england germany berlin".split())
    # model.doesnt_match("paris berlin london austria".split())
    # model.most_similar("man")
    # model.most_similar("queen")
    # model.most_similar("awful")
