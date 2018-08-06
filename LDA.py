__author__ = 'aalibasic'

import pandas as pd
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from collections import defaultdict
from pylab import *
import itertools
import re
import pickle
import csv

csv.field_size_limit(500 * 1024 * 1024)

input_file = 'NaukriGulfITSoftwareJobs.csv'
file_save_weights_pickle = 'sigmoid_weights_'+input_file+'LDAmodel.pkl'
file_save_weights_excel = 'sigmoid_weights_'+input_file+'LDAmodel.xlsx'

def run():

    from gensim import corpora, models, matutils

    # LOAD JOBS DATA
    jobsData = pd.read_csv(input_file)
    jobs_titles = jobsData['Title']
    jobs_descriptions = jobsData['Description']

    #ONET DATA (we can create new one with python script CreatingOnetDatabase.py
    allDat_ONET = pickle.load( open( "allDat_ONET.pkl", "rb" ) )
    # 1110 x 3 (O*NET-SOC Code	Title	Description)

    ############## Building Models #############################################################
    #We can create new dictionary with python script CreatingDictionary.py
    #load ONET dictionary
    dictionary = corpora.Dictionary.load('onet2018.dict')
    #load ONET corpus
    corpus = corpora.MmCorpus('onet2018.mm')

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

    #manual stoplist for all job datasets
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

    ##################################
    #END Pre-processing the data END #
    ##################################

    #more on models below: http://radimrehurek.com/gensim/tut2.html
    ###################################################################################################################
    #Tf-Idf Model Transforming vectors
    #tfidf is used to convert any vector from the old representation (bag-of-words integer counts) to the new representation (TfIdf real-valued weights):

    tfidf = models.TfidfModel(corpus, normalize=True) # step 1 -- initialize a model #normalize the resulting vectors to (Euclidean) unit length
    corpus_tfidf = tfidf[corpus] # step 2 -- use the model to transform vectors

    #########---LDA Model---########Latent Dirichlet Allocation, LDA##################
    # if you want to create a new model
    #lda = models.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=47, alpha='auto', eta='auto',  update_every=0, passes=5) # initialize an LSI transformation
    #lda.save('LSI_files/NewmodelLDA_2018_Armin.lda')
    # or use old model
    lsa=models.LdaModel.load('LSI_files/NewmodelLDA_2018_Armin.lda') #NewmodelLDA_2017_Armin

    # THIS IS CORPUS FROM DATA MINED JOBS
    vec_bow_jobs= [dictionary.doc2bow(text) for text in texts] #bow_corpus #THIS IS JOBS CORPUS
    lsa.update(vec_bow_jobs)
    vec_lsa = lsa[vec_bow_jobs]

    #LDA SIMILARITIES
    # ### Hellinger distance #lower means it is more similar
    from numpy import zeros
    _SQRT2 = np.sqrt(2)
    x = len(lsa[vec_bow_jobs])  # number of data mined jobs
    y = len(lsa[corpus])  # number of onet jobs
    simnew = zeros((x, y), dtype=float)
    for i in range(0, x):
        for j in range(0, y):
            dense1 = matutils.sparse2full(lsa[vec_bow_jobs[i]], lsa.num_topics)
            dense2 = matutils.sparse2full(lsa[corpus[j]], lsa.num_topics)
            sim = np.sqrt(np.sum((np.sqrt(dense1) - np.sqrt(dense2)) ** 2)) / _SQRT2
            simnew[i, j] = 1-sim
        print i
    sims = pd.DataFrame(simnew).transpose()

    #Standardization Armin applied both feature scaling and mean normalization
    sims = (sims - sims.mean()) / sims.std()

    #sum all weights to get one weight score for each job
    weights=sims.sum(axis=1)

    # Save occupations' weights
    weights=pd.DataFrame(np.hstack((pd.DataFrame(allDat_ONET['O*NET-SOC Code']), pd.DataFrame(weights))))
    with open(file_save_weights_pickle, 'wb') as handle:
      pickle.dump(weights, handle)

    jd=pd.read_csv('db_20_0/Occupation Data.txt','\t')
    desc_weights=pd.DataFrame(weights.sort_values(1, ascending=False)[:])
    desc_weights=pd.DataFrame(pd.merge(desc_weights, jd, left_on=0, right_on='O*NET-SOC Code', how='inner'),columns=['O*NET-SOC Code', 'Title'])

    writer = pd.ExcelWriter(file_save_weights_excel)
    desc_weights.to_excel(writer,'Sheet1')
    writer.save()

#LDA EVALUATION
vis = pyLDAvis.gensim.prepare(lda, corpus, dictionary, n_jobs=1)
pyLDAvis.save_html(vis, 'LDATopics43.html')
pyLDAvis.display(vis)

    # split into train and test - random sample, but preserving order
train_size = int(round(len(corpus) * 0.8))
train_index = sorted(random.sample(xrange(len(corpus)), train_size))
test_index = sorted(set(xrange(len(corpus))) - set(train_index))
train_corpus = [corpus[i] for i in train_index]
test_corpus = [corpus[j] for j in test_index]
perplexity_value = []
logperp = []
coherence_values = []
x = []

number_of_words = sum(cnt for document in test_corpus for _, cnt in document)
parameter_list = range(20, 80, 3)
for parameter_value in parameter_list:
print "starting pass for parameter_value = %.3f" % parameter_value
model = models.LdaModel(corpus=train_corpus, id2word=dictionary, num_topics=parameter_value, alpha=0.01, eta=0.01, iterations=5)

log_perplex = model.log_perplexity(test_corpus)
perplex = np.exp2(-log_perplex)
print "Total Perplexity: %s" % perplex
x.append(parameter_value)
perplexity_value.append(perplex)
logperp.append(log_perplex)

coherencemodel = CoherenceModel(model=model, corpus=train_corpus, coherence='u_mass')
coherence_values.append(coherencemodel.get_coherence())

# Show graph
x = np.array (x)
y = np.array (perplexity_value)
plt.plot(x, y)
plt.xlabel("Num Topics")
plt.ylabel("Perplexity Value")
plt.show()

y = np.array (logperp)
plt.plot(x, y)
plt.xlabel("Num Topics")
plt.ylabel("Log Perplexity")
plt.show()

y = np.array(coherence_values)
plt.plot(x, y)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.show()
