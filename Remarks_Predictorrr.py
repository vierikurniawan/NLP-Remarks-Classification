# import streamlit as st
import pandas as pd
import numpy as np
import pickle
import string
import nltk
import re

# Naive Bayes' package
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# NLP staple's packages
from nltk.corpus import stopwords
from nltk.corpus.reader.twitter import TweetTokenizer
from Sastrawi.Stemmer.Stemmer import Stemmer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import CountVectorizer

# Required Classes
class TextCleaning:
    def __init__(self, glossary, stopwords):
        self.glossary = dict(zip(glossary['slang'], glossary['formal']))
        self.stopwords = list(stopwords)
        self.factory = StemmerFactory()
        self.stemmer = self.factory.create_stemmer()
        self.tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)

    def clean_text(self, text):
        #tokenize tweet

        text_tokens = self.tokenizer.tokenize(text)

        text_clean = []
        for word in text_tokens:
          if (word not in self.stopwords and
              word not in string.punctuation):
            stem_word = self.stemmer.stem(word)
            text_clean.append(stem_word)
        return text_clean

    #remove punct
    def remove_punct(self, text_list):
      text = " ".join([char for char in text_list if char not in string.punctuation])
      return text

        #Fungsi Prepocessing

    def preprocess(self, text):
      text = text.lower() # lowercasing
      text = re.sub(r'[^0-9a-zA-Z]+', ' ', text) # non-alpha numeric
      text = re.sub(r'[\d+]+', '', text) # numeric
      text = re.sub(r'\\x[0-9a-fA-F]{2}', '', text) # emoji
      text = ' '.join([self.glossary[word] if word in self.glossary else word for word in text.split()]) # normalize
      text = ' '.join(['' if word in self.stopwords else word for word in text.split()]) # stopwords
      text = self.remove_punct(self.clean_text(text)) # punctuation
      return text


class CreateVocabulary:
    def __init__(self, dataset, preprocessor):
        self.dataset = dataset
        self.preprocessor = preprocessor

    def create_corpus(self):
        if type(self.dataset) == pd.core.frame.DataFrame:
          return list(self.dataset['remarks'])
        elif type(self.dataset) == list:
          return list(pd.concat(self.dataset)['remarks'])
        else:
          return None


    def create_vocabulary(self):
        corpus = self.create_corpus()
        vocabulary_list = [self.preprocessor.preprocess(text).split() for text in corpus]
        vocabulary = sum(vocabulary_list, [])
        vocabulary = list(set(vocabulary))
        vocabulary.sort()
        vocabulary = dict(zip(vocabulary, range(len(vocabulary))))
        return vocabulary


class RemarksPredictor:
    def __init__(self,model_option):
        if model_option == "MultinomialNB":
            self.model_version = 'MultinomialNB_ver1.1.pkl'
        elif model_option == "SVM":
           self.model_version = 'SVM_ver1.1.pkl'
        elif model_option == "RF":
           self.model_version = 'RFC_ver1.1.pkl'
        else:
           self.model_version = None

        

    def load_model(self):
        with open(self.model_version, 'rb') as model_file:
            self.model_pickle = pickle.load(model_file)
        return self.model_pickle

    def predict(self, text):
    #   stopwords = pd.read()
    #   preprocessor = TextCleaning
      model = self.load_model()
      return model.predict(text)

class DataFramePredictor:
    def __init__(self, nb_path='MultinomialNB_ver1.1.pkl', svm_path='SVM_ver1.1.pkl', rfc_path='RFC_ver1.1.pkl'):
        with open(nb_path, 'rb') as f:
            self.nb_model = pickle.load(f)
        with open(svm_path, 'rb') as f:
            self.svm_model = pickle.load(f)
        with open(rfc_path, 'rb') as f:
            self.rfc_model = pickle.load(f)
    def dfpredict(self, text):
        text['nb_pred'] = self.nb_model.predict(text['remarks'])
        text['svm_pred'] = self.svm_model.predict(text['remarks'])
        text['rfc_pred'] = self.rfc_model.predict(text['remarks'])
        text['final_pred'] = text.apply(self.model_polling, axis=1)
        return text

    def model_polling(self, text):
        nb_pred = text['nb_pred']
        svm_pred = text['svm_pred']
        rfc_pred = text['rfc_pred']

        if nb_pred == 'unknown' and svm_pred == 'unknown' and rfc_pred != 'unknown':
            return rfc_pred
        elif nb_pred == 'unknown' and rfc_pred == 'unknown' and svm_pred != 'unknown':
            return svm_pred
        elif svm_pred == 'unknown' and rfc_pred == 'unknown' and nb_pred != 'unknown':
            return nb_pred

        if nb_pred == svm_pred == rfc_pred:
            return nb_pred  # Semua prediksi sama
        elif nb_pred == svm_pred:
            return nb_pred  # NB dan SVM sama
        elif nb_pred == rfc_pred:
            return nb_pred  # NB dan RFC sama
        elif svm_pred == rfc_pred:
            return svm_pred  # SVM dan RFC sama
        else:
            return rfc_pred  # Semua berbeda, pilih RFC
        
def main(model_option):
   predictor = RemarksPredictor(model_option)
   text = ["rumah ortunya sudah didatangi, pembayaran partial via alfamidi telah dilakukan"]
   print(predictor.predict(text=text))


if __name__ == '__main__':
    model_option = "MultinomialNB"
    # model_option = "SVM"
    # model_option = "RF"
    main(model_option)