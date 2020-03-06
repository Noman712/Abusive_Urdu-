# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 17:20:44 2019

@author: Noman
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 12:45:52 2019

@author: Noman
"""


import numpy as np
import pandas as pd
# Import adjustText, initialize list of texts
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split

#Seed Random Numbers with the TensorFlow Backend
from numpy.random import seed
seed(1)

import warnings
warnings.filterwarnings("ignore")

class Urdu_Dataset:

    def avg_word(sentence):
      words = sentence.split()
      return (sum(len(word) for word in words)/len(words))
  
    def Generate_Urdu_Ngrams(self, _ngram_range=(1,1), _max_features=5000, words=True):
        
        
        #dataset_path = "F:\Machine_Learning\\Basic_Models\\maaz_data_set\\abusive_language_recognation_dataset.csv"
        dataset_path = "../dataset/abusive_language_recognation_dataset.csv"
    
        #load data
        df = pd.read_csv(dataset_path, usecols=['Tweets', 'Positive', 'Negative'])
        #print (df.head())
        #print(df.shape)
        rows,col = df.shape

        df['word_count'] = df['Tweets'].apply(lambda x: len(str(x).split(" ")))
        df[['Tweets','word_count']].head()
        
        df['char_count'] = df['Tweets'].str.len() ## this also includes spaces
        df[['Tweets','char_count']].head()
        
        
        df['avg_word'] = df['Tweets'].apply(lambda x:  Urdu_Dataset.avg_word(x))
        df[['Tweets','avg_word']].head()
        
        
        #print(df.describe())
        #print(df.sum(axis=0) )
        
        
        yeswisedf = df[(df['Positive'] == 'yes')]
        #print(yeswisedf.head())
        
        
        
        yeswisedf['word_count'] = yeswisedf['Tweets'].apply(lambda x: len(str(x).split(" ")))
        yeswisedf[['Tweets','word_count']].head()
        
        yeswisedf['char_count'] = yeswisedf['Tweets'].str.len() ## this also includes spaces
        yeswisedf[['Tweets','char_count']].head()
        
        
        yeswisedf['avg_word'] = yeswisedf['Tweets'].apply(lambda x:  Urdu_Dataset.avg_word(x))
        yeswisedf[['Tweets','avg_word']].head()
        
        
        #print(yeswisedf.describe())
        #print(yeswisedf.sum(axis=0) )
        
        
        
        nowisedf = df.loc[df['Negative'] == 'no']
        #print(nowisedf.head())
        
        
        
        nowisedf['word_count'] = nowisedf['Tweets'].apply(lambda x: len(str(x).split(" ")))
        nowisedf[['Tweets','word_count']].head()
        
        nowisedf['char_count'] = nowisedf['Tweets'].str.len() ## this also includes spaces
        nowisedf[['Tweets','char_count']].head()
        
        
        nowisedf['avg_word'] = nowisedf['Tweets'].apply(lambda x:  Urdu_Dataset.avg_word(x))
        nowisedf[['Tweets','avg_word']].head()
        
        
        #print(nowisedf.describe())
        #print(nowisedf.sum(axis=0) )
        
        
        #nowisedf = nowisedf[:3000]
        
        
        df_row = pd.concat([yeswisedf['Tweets'], nowisedf['Tweets']])
        #print(df_row.head())
        
        
        #print("Total Dataset:" + str(len(df_row)))
        #print("yes class:" + str(len(yeswisedf)))
        #print("no class:" + str(len(nowisedf)))
        
        

        Number_OF_Documents = rows + 1
        #Number_OF_Documents = len(yeswisedf) + len(nowisedf)
        Number_OF_POSITIVE_SAMPLES = len(yeswisedf)
        Number_OF_NEGATIVE_SAMPLES = len(nowisedf)
        
        
        
        y_train = np.empty(Number_OF_Documents-1)
       
        for i in range(0,Number_OF_Documents-1):
            if i < Number_OF_POSITIVE_SAMPLES - 1:
                y_train[i] = 1
            else:
                y_train[i] = 0
        
        
        i = 0
        #for index,row in df[:Number_OF_Documents].iterrows():
        for index,row in df.iterrows():
            positive = row['Positive']
            negative = row['Negative']
        
            
            if positive == "yes":
                y_train[i] = 1
            elif negative == "no":
                y_train[i] = 0
            else:
                y_train[i] = 0
            i = i + 1
            
            
        
        
        keywords_dictionary = []
        sentences_corpus = []
        #for index,row in df.iterrows(): 
        for index,row in df[:Number_OF_Documents].iterrows():
                text = str(row['Tweets'])
                
                sentences_corpus.append(text)
                list_of_words = text.split(" ")
                keywords_dictionary.append(list_of_words)
            
        
        #print("len(keywords_dictionary) {}".format(len(keywords_dictionary)))
        #print("keywords_dictionary[0] = {}".format(keywords_dictionary[0]))
        print("len sentences_corpus {}".format(len(sentences_corpus)))
        print(sentences_corpus[1])
        
        
        vocab = []
        vocab2 = []
        for kl in keywords_dictionary: 
            for w in kl:
                vocab.append(str(w))
                vocab2.append(str(w).lower())
        print("vocab len: {}".format(len(vocab)))
        print(vocab[:10])
        #print("vocab set len: {}".format(len(set(vocab))))
        #print("vocab2 len: {}".format(len(vocab2)))
        #print("vocab2 set len: {}".format(len(set(vocab2))))

        
        corpus = []
        for kl in sentences_corpus: 
            corpus = corpus + kl.split(',')

        print("len corpus {}".format(len(corpus)))
        print(corpus[-1])

        display_max = 200
        if words:
            # from vocab
            vectorizer = CountVectorizer(ngram_range=_ngram_range)
            Count_Vect = vectorizer.fit_transform(vocab)
            model_vocab_len = len(vectorizer.get_feature_names())
            print("Model feature # from *vocab* {}".format(model_vocab_len))
            #if model_vocab_len < _max_features:
            #    print("Model feature # {}".format(model_vocab_len))
            vectorizer = CountVectorizer(ngram_range=_ngram_range, max_features=_max_features)
            Count_Vect = vectorizer.fit_transform(vocab)
            model_vocab_len = len(vectorizer.get_feature_names())
            print("Model feature w/ max_feat *vocab* # {}".format(model_vocab_len))
            print(vectorizer.get_feature_names()[:display_max])
            #print(X.toarray())

            # from corpus
            vectorizer = CountVectorizer(ngram_range=_ngram_range)
            Count_Vect = vectorizer.fit_transform(corpus)
            model_vocab_len = len(vectorizer.get_feature_names())
            print("Model feature # from *corpus* {}".format(model_vocab_len))
            # if model_vocab_len < _max_features:
            #    print("Model feature # {}".format(model_vocab_len))
            vectorizer = CountVectorizer(ngram_range=_ngram_range, max_features=_max_features)
            X_count = vectorizer.fit_transform(corpus).toarray()
            model_vocab_len = len(vectorizer.get_feature_names())
            print("Model feature w/ max_feat *corpus* # {}".format(model_vocab_len))
            print(vectorizer.get_feature_names()[:display_max])
            # print(X.toarray())


            #Tf-iDF part
            vectorizer = TfidfVectorizer(ngram_range=_ngram_range)
            X_tfidf = vectorizer.fit_transform(corpus)
            print("TF-iDF Model feature # {}".format(len(vectorizer.get_feature_names())))
            print(vectorizer.get_feature_names()[:display_max])

            vectorizer = TfidfVectorizer(ngram_range=_ngram_range, max_features=_max_features)
            X_tfidf = vectorizer.fit_transform(corpus).toarray()
            print("TF-iDF Model feature # {}".format(len(vectorizer.get_feature_names())))
            print(vectorizer.get_feature_names()[:display_max])
        else:
            vectorizer = CountVectorizer(ngram_range=_ngram_range, token_pattern = r"(?u)\b\w+\b",  analyzer='char') #  AZ - Regular expression denoting what constitutes a “token”, only used if analyzer == 'word'.
            Count_Vect = vectorizer.fit_transform(vocab)
            model_vocab_len = len(vectorizer.get_feature_names())
            print("Model feature # {}".format(model_vocab_len))
            print(vectorizer.get_feature_names()[:display_max])

            vectorizer = CountVectorizer(ngram_range=_ngram_range, max_features=_max_features, analyzer='char')
            Count_Vect = vectorizer.fit_transform(vocab)
            model_vocab_len = len(vectorizer.get_feature_names())
            print("Model feature w/ max_feat *vocab* # {}".format(model_vocab_len))
            print(vectorizer.get_feature_names()[:display_max])


            # from corpus
            vectorizer = CountVectorizer(ngram_range=_ngram_range, analyzer='char')
            Count_Vect = vectorizer.fit_transform(corpus)
            model_vocab_len = len(vectorizer.get_feature_names())
            print("Model feature # from *corpus* {}".format(model_vocab_len))
            # if model_vocab_len < _max_features:
            #    print("Model feature # {}".format(model_vocab_len))
            vectorizer = CountVectorizer(ngram_range=_ngram_range, max_features=_max_features, analyzer='char')
            X_count = vectorizer.fit_transform(corpus).toarray()
            model_vocab_len = len(vectorizer.get_feature_names())
            print("Model feature w/ max_feat *corpus* # {}".format(model_vocab_len))
            print(vectorizer.get_feature_names()[:display_max])
            # print(X.toarray())

            #TfIDf
            vectorizer = TfidfVectorizer(ngram_range=_ngram_range, analyzer='char')
            X_tfidf = vectorizer.fit_transform(corpus).toarray()
            print("TF-iDF Model feature # {}".format(len(vectorizer.get_feature_names())))
            print(vectorizer.get_feature_names()[:display_max])

            vectorizer = TfidfVectorizer(ngram_range=_ngram_range,max_features=_max_features, analyzer='char') # You can still specify n-grams here.
            X_tfidf = vectorizer.fit_transform(corpus).toarray()
            print("TF-iDF Model feature with max-feat # {}".format(len(vectorizer.get_feature_names())))
            print(vectorizer.get_feature_names()[:display_max])
            

            
        print( "Shape of Tf-iDf final Ngram vector:" + str(X_tfidf.shape))
        print( "Shape of labels:" + str(y_train.shape))
        print(X_tfidf[2, :])

        print("Shape of Count final Ngram vector:" + str(X_count.shape))
        print("Shape of labels:" + str(y_train.shape))
        print(X_count[2, :200])

        #xTrain, xTest, yTrain, yTest = train_test_split(X_count, y_train, test_size = 0.2)
        xTrain, xTest, yTrain, yTest = train_test_split(X_tfidf, y_train, test_size=0.2)
        return xTrain, xTest, yTrain, yTest, sentences_corpus, keywords_dictionary, y_train
    


def main():
    dsurdu = Urdu_Dataset()
    #xTrain, xTest, yTrain, yTest, sentences_corpus, keywords_dictionary, y_train = ds.Generate_Urdu_Ngrams()


    print('# unigram char')
    xTrain1, xTest1, yTrain1, yTest1, sentences_corpus, keywords_dictionary, labels = dsurdu.Generate_Urdu_Ngrams(
        _ngram_range=(1, 1), _max_features=3000, words=False)
    print("keywords_dictionary {}".format(len(keywords_dictionary)))


    print('\n# bigram char')
    xTrain2, xTest2, yTrain2, yTest2, sentences_corpus, keywords_dictionary, labels = dsurdu.Generate_Urdu_Ngrams(
        _ngram_range=(2, 2), _max_features=3000, words=False)

    print('\n# trigram char')
    xTrain3, xTest3, yTrain3, yTest3, sentences_corpus, keywords_dictionary, labels = dsurdu.Generate_Urdu_Ngrams(
        _ngram_range=(3, 3), _max_features=3000, words=False)

    print('\n# uni+bigram char')
    xTrain12, xTest12, yTrain12, yTest12, sentences_corpus, keywords_dictionary, labels = dsurdu.Generate_Urdu_Ngrams(
        _ngram_range=(1, 2), _max_features=3000, words=False)

    print('\n#uni+trigram char')
    xTrain13, xTest13, yTrain13, yTest13, sentences_corpus, keywords_dictionary, labels = dsurdu.Generate_Urdu_Ngrams(
        _ngram_range=(1,3), _max_features=3000, words= False)

    print('\n# unigram word')
    wxTrain1, wxTest1, wyTrain1, wyTest1, sentences_corpus, keywords_dictionary, labels = dsurdu.Generate_Urdu_Ngrams(
        _ngram_range=(1, 1), _max_features=3000, words=True)

    print('\n# bigram word')
    wxTrain2, wxTest2, wyTrain2, wyTest2, sentences_corpus, keywords_dictionary, labels = dsurdu.Generate_Urdu_Ngrams(
        _ngram_range=(2, 2), _max_features=3000, words=True)



    print('#trigram word')
    wxTrain3, wxTest3, wyTrain3, wyTest3, sentences_corpus, keywords_dictionary, labels = dsurdu.Generate_Urdu_Ngrams(
        _ngram_range=(3,3), _max_features=3000, words= True)

    print('\n# uni+bigram word')
    wxTrain12, wxTest12, wyTrain12, wyTest12, sentences_corpus, keywords_dictionary, labels = dsurdu.Generate_Urdu_Ngrams(
        _ngram_range=(1, 2), _max_features=3000, words=True)

    print('\n# uni+trigram word')
    wxTrain13, wxTest13, wyTrain13, wyTest13, sentences_corpus, keywords_dictionary, labels = dsurdu.Generate_Urdu_Ngrams(
        _ngram_range=(1, 3), _max_features=3000, words=True)



if __name__ == "__main__":

    main()
