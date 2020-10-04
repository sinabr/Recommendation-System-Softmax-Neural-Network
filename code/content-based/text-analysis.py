from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.test.utils import common_dictionary, common_corpus  , get_tmpfile
from gensim.models import LsiModel
import gensim

import json
import nltk
import pandas

# sample_genre = pandas.read_csv('genre.csv',columns=['genre','movieId'])

#  Sample Movie Title
sample_data = ['silence of the lambs',
            'once upon a day in america']


token_list = word_tokenize(sample_data[0])
print(token_list)

tokens = [token_list]


#  LsiModel : Latent Semantic Analysis
from gensim import corpora

dictionary = corpora.Dictionary(tokens)
print(dictionary)
bow_corpus = [dictionary.doc2bow(text) for text in tokens]
model = LsiModel(common_corpus, id2word=common_dictionary)
vectorized_corpus = model[common_corpus]
fname = get_tmpfile('LSA.model')
model.save(fname)

# Term Frequncy Inverse Document Frequency

from gensim import models

tfidf = models.TfidfModel(bow_corpus)

# sample movie title
m_title = "silence of the lambs"


# NO RESULT 
words = m_title.lower().split(' ')
print(tfidf[dictionary.doc2bow(words)])