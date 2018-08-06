__author__ = 'aalibasic'

from gensim import corpora, models, similarities, matutils
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from collections import defaultdict
import pickle
import re
import logging
from pandas import DataFrame, read_csv
from numpy import array, savetxt
import pylab as plt
import numpy as np
import matplotlib.pyplot
import math

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def run():

#ONET DATA (we can create new one with python script CreatingOnetDatabase.py
allDat_ONET = pickle.load( open( "allDat_ONET.pkl", "rb" ) )

######  Create dictionary   ###########
# Creating the corpus and the dictionary

# remove common words, stemming, punctuation  and tokenize

documents=[]
expressions=['in order to', 'Check to ensure', 'All other']
for row in allDat_ONET['Description']:
for exp in expressions:
row=re.sub(exp,'', row)
documents.append(row)

#tokenize
tokenizer = RegexpTokenizer(r'\w+')
texts = [tokenizer.tokenize(document.lower())
for document in documents]
with open('tokenizer.pkl', 'wb') as handle:
pickle.dump(tokenizer, handle)

# remove words that appear only once
frequency = defaultdict(int)
for text in texts:
for token in text:
frequency[token] += 1

texts = [[token for token in text if frequency[token] > 1] for text in texts]

# #Plot the frequency before
# df = DataFrame() #create empty dataframe
# for w in frequency:
#     #print w, frequency[w]
#     df = df.append({'Word': w, 'Frequency': frequency[w]}, ignore_index=True)
# df.to_excel('wordsOnet2018Before.xlsx')
# ax = df.plot.line()
# ax.axhline(y=df['Frequency'].mean(), xmin=-1, xmax=1, color='r', linestyle='--', lw=2, label='Mean')
# ax.axhline(y=math.sqrt(df['Frequency'].var()), xmin=-1, xmax=1, color='g', linestyle='--', lw=2, label='Standard Deviation')
# plt.rcParams.update({'font.size': 22})
# plt.legend(loc='upper left')
# plt.xlabel("Words")
# plt.ylabel("Frequency")
# plt.show()


# English stoplist
stoplist = stopwords.words('english')
texts = [[word for word in text if word not in stoplist]
for text in texts]

# Manual stoplist
stoplist_manual = set('may usually separately often whose includes including primarily purposes necessary '
'using Using use Use ensure provide perform new according'.split())
texts = [[word for word in text if word not in stoplist_manual]
for text in texts]

# Stemming
stemmer = PorterStemmer()

def stem_tokens(tokens, stemmer):
stemmed_texts = []
for text in texts:
stemmed=[]
for token in text:
stemmed.append(stemmer.stem(token))
stemmed_texts.append(stemmed)
return stemmed_texts

texts = stem_tokens(texts, stemmer)

# Create dictionary and Corpus and save them for later use
# mapping between the words from description and documents (jobs)
dictionary = corpora.Dictionary(texts)
# additional filtering removing words that appear in more than 90% of the occupations and less than 2 occupations
dictionary.filter_extremes(no_below=2, no_above=0.9)
dictionary.save('onet2018.dict') #1110 documents (jobs) x 4662 unique words

#print("Unique total words from 1110 job description: ", dictionary.items())
# #4662 integer id assigned to each unique word
#print dictionary.get(0) #first word in dictionary

#To actually convert tokenized documents to vectors:
corpus = [dictionary.doc2bow(text) for text in texts] #list of 1110 jobs x matching words with unique words and counting them
# The function doc2bow() simply counts the number of occurrences of each distinct word,
# converts the word to its integer word id (from the dictionary) and returns the result as a sparse vector.
# Save corpus - result is vector idword, number that word appears in particular job
corpora.MmCorpus.serialize('onet2018.mm', corpus)
#print (len(corpus[1]))



# load ONET dictionary
dictionary = corpora.Dictionary.load('onet2018.dict')
# load ONET corpus
corpus = corpora.MmCorpus('onet2018.mm')

# Plot the frequency after
# df = DataFrame()  # create empty dataframe
# for cp in corpus:
#     for id, freq in cp:
#         df = df.append ({'Word': dictionary[id], 'Frequency': freq}, ignore_index=True)
# df = df.groupby('Word', as_index=False).sum()
# df = df.reset_index()
# df.to_excel('wordsOnet2018After.xlsx') # open excel, delete index and change file to csv

# df = read_csv("wordsOnet2018After.csv")
# ax = df.plot.line()
# ax.axhline(y=df['Frequency'].mean(), xmin=-1, xmax=1, color='r', linestyle='--', lw=2, label='Mean')
# ax.axhline(y=math.sqrt(df['Frequency'].var()), xmin=-1, xmax=1, color='g', linestyle='--', lw=2, label='Standard Deviation')
# plt.rcParams.update({'font.size': 22})
# plt.legend(loc='upper right')
# plt.xlabel("Words")
# plt.ylabel("Frequency")
# plt.show()

# # Get the data in matrix form
# # More info here https://radimrehurek.com/gensim/matutils.html#gensim.matutils.corpus2dense
# matrixCorpus = matutils.corpus2dense(corpus,  num_terms=4655)
# print(matrixCorpus.shape)
# #Documents are columns, that is why we use transpose so we have documents as rows
# df = DataFrame(matrixCorpus.transpose())
# df.to_csv("FullCorpusMatrixOnlyData2018.csv",header=None, index=None)
