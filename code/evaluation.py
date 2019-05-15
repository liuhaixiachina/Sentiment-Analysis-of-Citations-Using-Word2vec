import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn import svm, datasets
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn import model_selection
from sklearn.metrics import f1_score
from scipy import stats
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn import preprocessing
from patsy import dmatrices
import pandas as pd
import os
from nltk.corpus import stopwords
import nltk.data
import logging
import numpy as np  # Make sure that numpy is imported
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Imputer
from KaggleWord2VecUtility import KaggleWord2VecUtility
from sklearn.metrics import accuracy_score
import sys

reload(sys)
sys.setdefaultencoding('utf8')

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
    counter = 0
    #
    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    #
    # Loop through the reviews
    for review in reviews:
       #
       # Print a status message every 1000th review
       # if counter%1000. == 0.:
       #     print "Review %d of %d" % (counter, len(reviews))
       #
       # Call the function (defined above) that makes average feature vectors
       reviewFeatureVecs[counter] = makeFeatureVec(review, model, \
           num_features)
       #
       # Increment the counter
       counter = counter + 1
    return reviewFeatureVecs


def getCleanReviews(reviews):
    clean_reviews = []
    for review in reviews["review"]:
        clean_reviews.append( KaggleWord2VecUtility.review_to_wordlist( review, remove_stopwords=True ))
    return clean_reviews



if __name__ == '__main__':
    full = pd.read_csv( os.path.join(os.path.dirname(__file__), "data/fortrainmodel/sentiment-n-p.csv"), header=0, quoting=3 )
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    # Set values for various parameters
    num_features = 300    # Word vector dimensionality
    # model = Word2Vec.load_word2vec_format('senti_specific', binary=False)
    #outsmalltest: out.txt
    # model = Word2Vec.load_word2vec_format(str(w2vmodel), binary=False)
    # print model #Word2Vec(vocab=550, size=50, alpha=0.025)
    # model_name = "trainedmodels/out_word2vec_model_acl.txt.vec"
    # model_name = "trained_brown_model"#
    # model_name = "pos_neg.txt-round-4"
    model_name = "trainedmodels/ACL300" #Word2Vec(vocab=13685, size=300, alpha=0.025)
    # model_name = "trainedmodels/out_mixedabs.txt.vec" #out_mixedabs.txt.vec(0.746), out_word2vec_model_acl(0.755)
    # model_name = "trainedmodels/MixedAbs300"
    # model_name = "trainedmodels/out_word2vec_model_acl.txt.vec" #out_word2vec_model_acl.txt.vec:Word2Vec(vocab=764751, size=100, alpha=0.025)
    # model_name = "brown"#(0.752)
    model = Word2Vec.load(model_name) #for binary format
    # model=Word2Vec.load_word2vec_format(model_name, binary=False) # for text format
    # print model
    fullDataVecs = getAvgFeatureVecs( getCleanReviews(full), model, num_features )
    # print fullDataVecs
    imp=Imputer(missing_values='NaN',strategy='mean',axis=0)
    imp.fit_transform(fullDataVecs)
    new_full_data=imp.transform(fullDataVecs)
    full_data_X = new_full_data
    print full_data_X
    y=full["sentiment"]

    desired_array = [int(numeric_string) for numeric_string in y]
    y=np.ravel(desired_array)
    # n_classes = y.shape[1]
    # print 'n_classese: '+str(n_classes)
# Split into training and test

    # haixia comment
#     n=0
#     summacrof=0
#     for i in range(0,1):
#         n=n+1
#         X_train, X_test, y_train, y_test = train_test_split(full_data_X, y, test_size=.1)
#         print X_train.shape
#         print y_train.shape
#         # print X_train
#         # Run classifier
#         # classifier = OneVsRestClassifier(svm.SVC(kernel='linear', class_weight='auto',probability=True))
#         # y_score = classifier.fit(X_train, y_train).decision_function(X_test)
#         # y_score = classifier.fit(X_train, y_train).predict(X_test)
#                                          #
#         # y_score = classifier.predict(X_test)
#         # y_score = model_selection.cross_val_score(classifier, full_data_X, y, cv=5)
#         # cv = model_selection.ShuffleSplit(n_classes, n_iter=3, test_size=0.3, random_state=0)
#         # y_score=model_selection.cross_val_score(classifier, full_data_X, y, cv=cv)
#         # print y_score
#
#         # Compute Precision-Recall and plot curve
#         precision = dict()
#         recall = dict()
#         average_precision = dict()
#         average_recall = dict()
#         macro_f_dic=dict()
#
#         # for i in range(n_classes):
#         #     print i
#         #     clf = svm.SVC(kernel='linear')
#         #     clf = svm.SVC(gamma=2, C=1,class_weight='auto')
#         #     clf.fit(X_train, y_train.ravel())
#         #     y_pred=clf.predict(X_test)
#         #     macro_f_dic[i]=f1_score(y_test[:, i], y_pred,average="macro")
#         #     print macro_f_dic[i]
#         #     print y_pred
#
#
#         #=======without mult class==============
#
#         # clf = svm.SVC(kernel='linear')
#         # clf = svm.SVC(gamma=2, C=1,class_weight='auto')
#         # # clf.fit(X_train, y_train.ravel())
#         # clf.fit(X_train, y_train)
#         # y_score=clf.predict(X_test)
#         y_score=OneVsRestClassifier(LinearSVC(random_state=0)).fit(X_train, y_train).predict(X_test)
#         # for yi in y_score:
#         #     print 'y_score: '+str(yi)
#
#         # print y_score
#         # lb = preprocessing.LabelBinarizer()
#         # lb.fit(y_test)
#         # macro_f=f1_score(y_test, y_score,average="macro")
#         macro_f=f1_score(y_test, y_score)
#         macro_f = accuracy_score(y_test, y_score)
#         print 'macro_f: '+str(macro_f)
#         summacrof=summacrof+macro_f
#         # =======without mult class==============
#     print n
#     print 'summacrof = '+str(summacrof/n)
# haixia comment

    # shuffle = model_selection.ShuffleSplit(len(y), n_iter=10)
    clf = LinearSVC(penalty = 'l2')
    # clf = svm.SVC(kernel='linear')
    n_folds=10
    # precision = model_selection.cross_val_score(clf, full_data_X, y, cv=n_folds, scoring='precision')
    # print('Precision', np.mean(precision), precision)
    #
    # recall = model_selection.cross_val_score(clf, full_data_X, y, cv=n_folds, scoring='recall')
    # print('Recall', np.mean(recall), recall)

    f1 = model_selection.cross_val_score(clf, full_data_X, y, cv=n_folds, scoring='f1')
    print('F1', np.mean(f1), f1)

    # # Compute micro-average ROC curve and ROC area
    # precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(),
    #     y_score.ravel())
    # average_precision["micro"] = average_precision_score(y_test, y_score,
    #                                                      average="micro")
    # # print 'micro_precision: '+str(average_precision["micro"])
    #
    # average_precision["macro"] = average_precision_score(y_test, y_score,
    #                                                      average="macro")
    # print 'macro_precision: '+str(average_precision["macro"])
    #
    #
    # average_recall["macro"] = average_precision_score(y_test, y_score,
    #                                                      average="macro")
    # print 'macro_precision: '+str(average_precision["macro"])

# Macro-average precision = (P1+P2)/2 = (57.14+68.49)/2 = 62.82
# Macro-average recall = (R1+R2)/2 = (80+84.75)/2 = 82.25
# The Macro-average F-Score will be simply the harmonic mean of these two figures.
# print stats.hmean([ -50.2 , 100.5 ])
# def hmean(*args):
#     return len(args) / sum(1. / val for val in args)
    #
    # # Plot Precision-Recall curve
    # plt.clf()
    # plt.plot(recall[0], precision[0], label='Precision-Recall curve')
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.ylim([0.0, 1.05])
    # plt.xlim([0.0, 1.0])
    # plt.title('Precision-Recall example: AUC={0:0.2f}'.format(average_precision[0]))
    # plt.legend(loc="lower left")
    # plt.show()
    #
    # # Plot Precision-Recall curve for each class
    # plt.clf()
    # plt.plot(recall["micro"], precision["micro"],
    #          label='micro-average Precision-recall curve (area = {0:0.2f})'
    #                ''.format(average_precision["micro"]))
    # for i in range(n_classes):
    #     plt.plot(recall[i], precision[i],
    #              label='Precision-recall curve of class {0} (area = {1:0.2f})'
    #                    ''.format(i, average_precision[i]))
    #
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.title('Extension of Precision-Recall curve to multi-class')
    # plt.legend(loc="lower right")
    # plt.show()
