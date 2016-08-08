
# coding: utf-8

# # What Language Am I Anyway
# Where the rules are naive and the scores don't matter.

# In[1]:

import numpy as np
import glob
import csv
import re
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer as tuff
from sklearn.pipeline import Pipeline
from extensions import extensions


# ## Reading and Storing Training Data

# In[2]:

def read_prog_files(loc):
    files = glob.glob(loc, recursive=True)
    texts = []
    for file in files:
        with open(file, encoding='latin_1') as f:
            texts.append(f.read())
    return texts


# ### Placing samples (for training) and their appropriate labels into lists at one time

# In[3]:

samples = []
labels = []

for ext, name in extensions.items():
    x = read_prog_files('bmgame/bmgame/bench/**/*.{}'.format(ext))
    samples += x
    y = len(x) * [name]
    labels += y


# ### Creating a pipeline with a CountVectorizer and Multinomial Naive Bayes Classifier

# In[5]:

pip = Pipeline([('cv', CountVectorizer(analyzer='word', token_pattern=r'[a-zA-Z]{2,}|\s{2,}|[^\w\d\s]')), ('bay', MultinomialNB())])


# ### Fitting my pipeline with samples (code snippet) and labels (code type)
# The score of samples to labels is as expected (close to 1)

# In[6]:

pip.fit(samples, labels)
pip.score(samples, labels)


# ### Creating a train_test_split to ensure a good set of data
# My test lists score fairly well (.87) and am feeling somewhat confident going forward with this fitting.

# In[7]:

train_X, test_x, train_y, test_y = train_test_split(samples, labels, train_size=.6, random_state=42)
pip.fit(train_X, train_y)
pip.score(test_x, test_y)


# ## Reading the Unknown Testing Samples

# In[8]:

unknown = []
ulabels = []
for item in range(1, 33):
    x = read_prog_files('test/{}'.format(item))
    unknown += x
with open('test.csv') as testy:
    reader = csv.reader(testy)
    for row in reader:
        ulabels.append(row[1])


# ### Scoring my unseen code snippets to the correct labels

# In[9]:

pip.score(unknown, ulabels)


# In[10]:

pip.predict(unknown)


# ## Why am I only getting around .75?
# Here I am counting my set of labels that I fitted my pipeline with... perhaps more sampling would improve my model... that usually seems to be the answer. 

# In[12]:

count_d = {}
for item in labels:
    count_d.setdefault(item, 0)
    count_d[item] += 1


# In[14]:

count_d


# ## Predict Function

# In[28]:

def get_language(code):
    return pip.predict([code])[0]


# In[29]:

get_language('numpy')


# In[ ]:



