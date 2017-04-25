# Sentiment-Analysis-of-Citations-Using-Word2vec

This project is for the paper: Sentiment Analysis of Citations Using Word2vec (https://arxiv.org/abs/1704.00177)


You may need to install the following packages: pandas, nltk, numpy, gensim and sklearn, as well as the source file for computing the vectors "KaggleWord2VecUtility" (https://github.com/wendykan/DeepLearningMovies/blob/master/KaggleWord2VecUtility.py)

1. To train the word2vec model, refer to the Python code: trainwordvects.py

2. To evaluate the model's effects on classifying the sentiment-sentences, run the file: evaluation.py

3. For P-S specific model training, please contact Dr. Duyu Tang for code ( http://ir.hit.edu.cn/~dytang/ ) 
There are already-trained models for use: under the folder "trainedmodels". Note that the model "PS-ACL300" is a text file. When you try to load the text format model, use this code: model=Word2Vec.load_word2vec_format(model_name, binary=False) # for text format

 
 Post experiment on using vectors trained on Googlenews
:
('F1', 0.38) for positive and negtive classification.
