__author__ = 'aalibasic'

import pandas as pd
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from collections import defaultdict
import itertools
#import enchant
from pylab import *
import re
import pickle
import sys
import csv

#csv.field_size_limit(sys.maxsize)
csv.field_size_limit(500 * 1024 * 1024)

def run():

from gensim import corpora, models, similarities

#LOAD JOBS DATA
jobsGCCOil = pickle.load( open( "jobsGCCOil.pkl", "rb" ) )
jobs_titles=jobsGCCOil[0]
jobs_descriptions=jobsGCCOil[1]
# 2685 jobs total for Oil & Gas Industry in Gulf Area

#ONET DATA (we can create new one with python script CreatingOnetDatabase.py
allDat_ONET = pickle.load( open( "allDat_ONET.pkl", "rb" ) )
# 1110 x 3 (O*NET-SOC Code	Title	Description)

############## Building Models #############################################################
#We can create new dictionary with python script CreatingDictionary.py
#load ONET dictionary
dictionary = corpora.Dictionary.load('onet.dict')
#load ONET corpus
corpus = corpora.MmCorpus('onet.mm')

###########################
# Pre-processing the data #
###########################

#We can create new tokenizer with python script CreatingDictionary.py
tokenizer = pickle.load( open( "tokenizer.pkl", "rb" ) )

# remove specific expressions
documents=[]
file = open('text_preprocessing_jobs/expressions_jobs.txt', 'r')
expressions=[]
lines=file.readlines()
for line in lines:
line=line.strip("\n").strip()
line= line.split(",")
expressions.append(line)
expressions=list(itertools.chain(*expressions))

for i in range(0, len(jobs_descriptions)):
row=jobs_descriptions[i].lower()
#row=toy_docs1[i].lower()
for exp in expressions:
exp=str(exp).strip(" '' ")
row = re.sub(exp, '', row).strip()
row = re.sub(r'\w*\d\w*', '', row).strip()
row=row.replace('\\t', ' ')
documents.append(row.replace('\t', ' '))

# remove words with specific prefixes
texts=[]
prefixes = ('www','detailswebsitehttp:', )
for document in documents:
d=document.split()
for word in document.split():
if word.startswith(prefixes):
d.remove(word)
texts.append(' '.join(d))

# remove common words, stemming, punctuation  and tokenize
#tokenize
texts = [tokenizer.tokenize(text)
for text in texts]

#manual stoplist by Ioannis fo all job datasets
file = open('text_preprocessing_jobs/stoplist_manual_jobs.txt', 'r')
stoplist_manual=[]
lines=file.readlines()
for line in lines:
stoplist_manual.append(line.rsplit())
stoplist_manual=list(itertools.chain(*stoplist_manual))
texts = [[word for word in text if word not in stoplist_manual]
for text in texts]

# common english stoplist
stoplist = stopwords.words('english')
texts = [[unicode(word, errors='replace') for word in text if unicode(word, errors='replace') not in stoplist]
for text in texts]

# remove words that appear only once
frequency = defaultdict(int)
for text in texts:
for token in text:
frequency[token] += 1

texts = [[token for token in text if frequency[token] > 1] #these two lines of code remove words that appear only once
for text in texts]
#print (frequency)
wordsOnet = []
for w in sorted(frequency, key=frequency.get, reverse=True):
#print w, frequency[w]
wordsOnet.append(w +" " + str(frequency[w]))
df = pd.DataFrame(wordsOnet)
df.to_excel('wordsGCCoil.xlsx', header=False, index=False)


# Remove empty strings
texts = [[word for word in text if word]
for text in texts]

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
# texts_held_out = texts[2000:] #we leave last 685 jobs for test evaluation
# texts = texts[:2000]

##################################
#END Pre-processing the data END #
##################################


#more on models below: http://radimrehurek.com/gensim/tut2.html
###############################################################
#Tf-Idf Model Transforming vectors
#tfidf is used to convert any vector from the old representation (bag-of-words integer counts) to the new representation (TfIdf real-valued weights):

tfidf = models.TfidfModel(corpus, normalize=True) # step 1 -- initialize a model #normalize the resulting vectors to (Euclidean) unit length
corpus_tfidf = tfidf[corpus] # step 2 -- use the model to transform vectors
# print (corpus_tfidf[0])
# print (corpus[0])

########---LSA Model---#####Latent Semantic Indexing, LSI (or sometimes LSA)################################
lsa = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=300) # initialize an LSI transformation
lsa.save('LSI_files/NewmodelLSA_2017_Armin.lsi')
#lsa=models.LsiModel.load('LSI_files/model_2015.lsa')
lsa=models.LsiModel.load('LSI_files/NewmodelLSA_2017_Armin.lsi')

### Cosine similarity
# transform corpus to LSI space and index it
index = similarities.MatrixSimilarity(lsa[corpus]) #corpus_tfidf was before #CHANGED
index.save('onet.index')
index = similarities.MatrixSimilarity.load('onet.index')

# THIS IS CORPUS FROM DATA MINED JOBS
vec_bow_jobs= [dictionary.doc2bow(text) for text in texts] #bow_corpus #THIS IS JOBS CORPUS
#vec_lsa = lsa[tfidf[vec_bow_jobs]] #THIS IS NOW TRANSFORMED CORPUS FOR JOBS # convert the jobs to LSI space
vec_lsa = lsa[vec_bow_jobs]
print (lsa.show_topics())

sims=[]
for similarities in index[vec_lsa]:
sims.append(similarities)

sims=pd.DataFrame(np.array(sims)).transpose()
print "sims posle", sims.shape
# writer = pd.ExcelWriter('SIMILARITIESBEFORE_2015_GCC_oil&gas.xlsx')
# sims.to_excel(writer,'Sheet1')
# #df2.to_excel(writer,'Sheet2')
# writer.save()

#Standardization Armin applied both feature scaling and mean normalization
sims = (sims - sims.mean()) / sims.std()

#we can also apply Min-Max scaling
#sims = (sims - sims.min()) / (sims.max() - sims.min())

# # change the range of the values #old Ioannis standardization
# sims -= sims.min()
# sims /= sims.max()
#
# #sigmoid function
# for i in range (0, len(sims)):
#     m=1.0
#     c=0.53
#     s=0.05
#     sims.iloc[i]= m/(1+np.exp((-(np.array(sims.iloc[i])-c))/s)) #ioannis
#     #sims.iloc[i]= 1/(1+np.exp(np.array(sims.iloc[i]))) #armin
#
# #normalization of weights #old Ioannis standardization
# sims/=sims.sum()
weights=sims.sum(axis=1)

weights=pd.DataFrame(np.hstack((pd.DataFrame(allDat_ONET['O*NET-SOC Code']), pd.DataFrame(weights))))
with open('sigmoid_weights_2015_GCC_oil&gas.pkl', 'wb') as handle:
pickle.dump(weights, handle)

# Visualize occupations' weights
jd=pd.read_csv('db_20_0/Occupation Data.txt','\t')
desc_weights=pd.DataFrame(weights.sort_values(1, ascending=False)[:])
desc_weights['Weight_pct']= (desc_weights[1]/desc_weights[1].sum())*100
desc_weights=pd.DataFrame(pd.merge(desc_weights, jd, left_on=0, right_on='O*NET-SOC Code', how='inner'),columns=[1,'O*NET-SOC Code','Title', 'Weight_pct'])

print 'Top 10 most demanded occupations for Oil & Gas in GCC countries'
print desc_weights.Title[:10]

writer = pd.ExcelWriter('sigmoid_weights_2015_GCC_oil&gas.xlsx')
desc_weights.to_excel(writer,'Sheet1')
#df2.to_excel(writer,'Sheet2')
writer.save()
