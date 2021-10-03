# -*- coding: utf-8 -*-
"""
Created on Fri May 22 17:11:04 2020

@author: Roshni, Raja Alfiq
"""

import pandas as pd
import numpy as np
import seaborn as sb
import re
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

#==============================================================================

# Extracting data from the csv file into mbti_dataset
mbti_dataset = pd.read_csv("mbti_1.csv")

# Previewing the general data & the first post
print("Data Preview : The first five rows of the MBTI dataset")
print(mbti_dataset.head())

print()
print("Data Preview : The first test subject's posts")
print()
for post in mbti_dataset.head(1).posts.values:
    print(post.split('|||'))
    print()

# type_count holds the total of entries of each type
type_count = mbti_dataset['type'].value_counts()
 # Visualising the data using a barchart
print()
print("Data Visualisation : Number of Occurences of each Personality Type")
plt.figure(figsize=(8,4))
barplt = sb.barplot(type_count.index, type_count.values, alpha= 0.8)
plt.ylabel('Frequency', fontsize =14)
plt.xlabel('Types', fontsize = 14)
plt.show()

#function that binarize the features
def types_ind (row) :
    r = row['type']

    I=0
    N=0
    T=0
    J=0

    if r[0] == 'I':
        I=1
    elif r[0] == 'E':
        I=0

    if r[1] == 'N':
        N=1
    elif r[1] == 'S':
        N=0

    if r[2] == 'T':
        T=1
    elif r[2] == 'F':
        T=0

    if r[3] == 'J':
        J=1
    elif r[3] == 'P':
        J=0

    return pd.Series({'IE':I, 'NS':N, 'TF':T, 'JP':J})

print()
print("Occurences of each personality")
print()
mbti_dataset = mbti_dataset.join(mbti_dataset.apply(lambda row: types_ind (row), axis= 1))
copy_mbti = mbti_dataset.copy()
#printing the data frame
del copy_mbti['posts']
print(copy_mbti.head())

print()
print()
#Holding the counts of a feature over the other
print("Introversion(I) / Extroversion(E) : ",
      mbti_dataset['IE'].value_counts()[0]," / ",mbti_dataset['IE'].value_counts()[1])
print("Intuition(N) / Sensing(S) : ",
      mbti_dataset['NS'].value_counts()[0]," / ",mbti_dataset['NS'].value_counts()[1])
print("Thinking(T) / Feeling(F) : ",
      mbti_dataset['TF'].value_counts()[0]," / ",mbti_dataset['TF'].value_counts()[1])
print("Judging(J) / Perceiving(P) : ",
      mbti_dataset['JP'].value_counts()[0]," / ",mbti_dataset['JP'].value_counts()[1])
# data visualisation of the percentage of each personality type using a pie chart
print()
print("Percentages of each personality")
labels=['INFP','INFJ','INTP','INTJ','ENTP','ENFP','ISTP','ISFP','ENTJ','ISTJ','ENFJ','ISFJ','ESTP','ESFP','ESFJ','ESTJ']
sizes= [21.1,16.9,15,12.6,7.9,7.78,3.88,3.12,2.66,2.36,2.19,1.91,1.03,0.553,0.484,0.45]

explode = (0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.4,0.4,0.4,0.4,0.4)
fig1, ax1 = plt.subplots()
plt.pie(sizes, explode=explode, labels=labels, pctdistance=0.6,autopct='%2.1f%%',
        shadow=True,labeldistance = 1.2,center=(-5,0),wedgeprops={'linewidth':6})

ax1.axis('equal')
plt.show()

# visualising the dataset using heat map through correlation
pfc = mbti_dataset[['IE','NS','TF','JP']].corr()
print(pfc.head())
print()
heat_map = plt.cm.RdBu
plt.figure(figsize=(5,5))
plt.title('Correlation of Features', size=15)
plt.show(sb.heatmap(pfc, cmap=heat_map,  annot=True,linecolor ='black'))

# binarizing personality types
bin_Pers = {'I':0, 'E':1, 'N':0, 'S':1, 'F':0, 'T':1, 'J':0, 'P':1}
bin_Pers_list = [{0:'I', 1:'E'}, {0:'N', 1:'S'}, {0:'F', 1:'T'}, {0:'J', 1:'P'}]

def biss_personality(pers) :
    return [bin_Pers[l]for l in pers]


def mbti_binarize(pers) :

    s = ""
    for i, l in enumerate(pers):
        s += bin_Pers_list[i][l]
    return s

bin_mbti = mbti_dataset.head(5)
list_pers_bin = np.array([biss_personality(p) for p in bin_mbti.type])
print("Binarized MBTI : \n%s" % list_pers_bin)
print()

##### Compute list of subject with Type | list of comments
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import word_tokenize

# We want to remove these from the psosts
unique_type_list = ['INFJ', 'ENTP', 'INTP', 'INTJ', 'ENTJ', 'ENFJ', 'INFP', 'ENFP',
       'ISFP', 'ISTP', 'ISFJ', 'ISTJ', 'ESTP', 'ESFP', 'ESTJ', 'ESFJ']

unique_type_list = [x.lower() for x in unique_type_list]

# Lemmatize
stemmer = PorterStemmer()
lemmatiser = WordNetLemmatizer()

# Cache the stop words for speed
cachedStopWords = stopwords.words("english")

def pre_process_data(mbti_dataset, remove_stop_words=True, remove_mbti_profiles=True):

    list_personality = []
    list_posts = []
    len_data = len(mbti_dataset)
    i=0

    for row in mbti_dataset.iterrows():
        i+=1
        if (i % 500 == 0 or i == 1 or i == len_data):
            print("%s of %s rows" % (i, len_data))

        ##### Remove and clean comments
        posts = row[1].posts
        temp = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', posts)
        temp = re.sub("[^a-zA-Z]", " ", temp)
        temp = re.sub(' +', ' ', temp).lower()
        if remove_stop_words:
            temp = " ".join([lemmatiser.lemmatize(w) for w in temp.split(' ') if w not in cachedStopWords])
        else:
            temp = " ".join([lemmatiser.lemmatize(w) for w in temp.split(' ')])

        if remove_mbti_profiles:
            for t in unique_type_list:
                temp = temp.replace(t,"")

        type_labelized = biss_personality(row[1].type)
        list_personality.append(type_labelized)
        list_posts.append(temp)

    list_posts = np.array(list_posts)
    list_personality = np.array(list_personality)
    return list_posts, list_personality

list_posts, list_personality  = pre_process_data(mbti_dataset, remove_stop_words=True)
print()
print("Num posts and personalities: ",  list_posts.shape, list_personality.shape)
print()
#list post is a vector, list personality is a 2D matrix 8765 rows and 4 columns

# Sklearn was used to recognize the words

# Posts to a matrix of token counts
cntizer = CountVectorizer(analyzer="word",
                             max_features=1500,
                             tokenizer=None,
                             preprocessor=None,
                             stop_words=None,
                             max_df=0.7,
                             min_df=0.1)

# Learn the vocabulary dictionary and return term-document matrix
X_cnt = cntizer.fit_transform(list_posts)

# Transform the count matrix to a normalized tf or tf-idf representation
tfizer = TfidfTransformer()

# Learn the idf vector (fit) and transform a count matrix to a tf-idf representation
X_tfidf =  tfizer.fit_transform(X_cnt).toarray()

# Creating a list of the feature names
feature_names = list(enumerate(cntizer.get_feature_names()))
feature_names

print(X_tfidf.shape)
print()
#8675 rows, 791 columns

personalityList = []

for i in mbti_dataset.iterrows():
    type_label = biss_personality(i[1].type)
    personalityList.append(type_label)
    
personalityList = np.array(personalityList)

print("X: Posts in tf-idf representation \n* 1st row:\n%s" % X_tfidf[0])
#===============================XGBOOST========================================
#Initialize X and Y
print("Training The Model")
X = X_tfidf

print("Accuracy For Each Personality Catergory")
mbti_type = [ "Introversion/ Extroversion (IE):", 
               "Intuition/ Sensing(NS):", 
               "Feeling/ Thinking(FT):", 
               "Judging/ Perceiving(JP):"]

for l in range(len(mbti_type)):
    
    Y = list_personality[:,l]

#split data into train and test sets, the training to testing ratio is 8:2
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, 
                                       random_state = 42)

#Training the model
    model = XGBClassifier()
    model.fit(X_train, y_train)

#predicting for test data
    y_pred = model.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print("%s %.2f%%" % (mbti_type[l], accuracy * 100.0))

print()
#Evaluate XGBoost Models with Learning Curve
#logloss evaluation metric for binary logarithmic loss
#Loss function is to return probability value on predictions (high - bad)
#This is a visual report of how the model is performing in the training and testing set
print("Monitoring Training Performance and Evaluating XGBoost Model")
print("Accuracy For Each Personality Catergory")
for l in range(len(mbti_type)):
    
    Y = list_personality[:,l]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, 
                                       random_state = 42)
    model = XGBClassifier()
    eval_set = [(X_train, y_train), (X_test, y_test)]
    model.fit(X_train, y_train, eval_metric = ["error", "logloss"], eval_set = eval_set, verbose = False)
    y_pred = model.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print("%s %.2f%%" % (mbti_type[l], accuracy * 100.0))
    #Retrieve performance metrics
    results = model.evals_result()
    epochs = len(results["validation_0"]["error"])
    x_axis = range(0, epochs)
    #plot log loss
    fig, ax = plt.subplots()
    ax.plot(x_axis, results["validation_0"]["logloss"], label = "Train")
    ax.plot(x_axis, results["validation_1"]["logloss"], label = "Test")
    ax.legend()
    plt.ylabel("Log Loss")
    plt.title("XGBoost Log Loss of " + mbti_type[l])
    plt.show()
    #plot classification error 
    fig, ax = plt.subplots()
    ax.plot(x_axis, results["validation_0"]["error"], label = "Train")
    ax.plot(x_axis, results["validation_1"]["error"], label = "Test")
    ax.legend()
    plt.ylabel("Classification Error")
    plt.title("XGBoost Classification Error " + mbti_type[l])
    plt.show()
    print("==================================================================")
    
#Using Early Stopping to avoid Overfitting
# error = binary classification error rate
# auc removes classification that are confident but wrong (false positive)
# fine tuning the model
print("Accuracy For Each Personality Catergory")
for l in range(len(mbti_type)):
    
    Y = list_personality[:,l]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, 
                                       random_state = 42)
    model = XGBClassifier()
    eval_set = [(X_test, y_test)]
    model.fit(X_train, y_train, eval_metric = ["logloss", "error", "auc"], eval_set = eval_set, verbose = True, early_stopping_rounds = 10)
    y_pred = model.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print("%s %.2f%%" % (mbti_type[l], accuracy * 100.0))
    print("==================================================================")
    
#Feature importance
#importance provides a score that indicates how useful or valuable each feature was in the construction of the boosted decision trees within 
#the model. The more an attribute is used to make key decisions with decision trees, the higher its relative importance.
Y = list_personality[:,0]
model = XGBClassifier()
model.fit(X,Y)
ax = plot_importance(model, max_num_features = 20)
fig = ax.figure
fig.set_size_inches(15,10)
plt.show()

#enumerate : adds a counter to an interable, returns enumerate object that can be used in loops or coverted to list
feature_list = sorted(list(enumerate(model.feature_importances_)), key = lambda x: x[1], reverse = True)
print("index    tf-idf        feature name")
for f in feature_list[0:25]:
    print("%d\t%f\t%s" % (f[0], f[1], cntizer.get_feature_names()[f[0]]))   

#Displaying XGBoost default parameters 
print()
default_param = model.get_xgb_params()
print(default_param)
print()

#Configuring Gradient Boosting 
print("Accuracy For Each Personality Category")
for l in range(len(mbti_type)):
    
    Y = list_personality[:, l]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)
    model = XGBClassifier(learning_rate = 0.1,
                      colsample_bytree = 0.4,
                      subsample = 0.8,
                      n_estimators = 200,
                      reg_alpha = 0.5,
                      max_depth = 4)
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    accuracy = metrics.accuracy_score(Y_test, y_pred)
    print("%s %.2f%%" % (mbti_type[l], accuracy * 100.0))

#XGBoost Hyperparameter Tuning
#k-fold is not suitable for imbalance data
print()
print("XGBoost Hyperparameter Tuning")
'''
X = X_tfidf

for l in range(len(mbti_type)):
    print(mbti_type[l])
    
    Y = list_personality[:,l]
    model = XGBClassifier(n_estimators = 200,
                          max_depth = 4,
                          learning_rate = 0.1,
                          colsample_bytree = 0.4,
                          subsample = 0.8,
                          reg_alpha = 0.5)
    param_range = {
            'n_estimators': [200,300],
            'learning_rate' : [0.1, 0.2],
            'max_depth': [4,5],
            }
    
    skf = StratifiedKFold( n_splits = 10, random_state = 42, shuffle = True)
    gridSearch = GridSearchCV(model, param_range, cv = skf, n_jobs = -1, scoring = "neg_log_loss")
    gridResult = gridSearch.fit(X, Y)
    
    print("* Best: %f using %s" % (gridResult.best_score_, gridResult.best_params_))
    means = gridResult.cv_results_['mean_test_score']
    stds = gridResult.cv_results_['std_test_score']
    params = gridResult.cv_results_['params']
    #zip function is to take iterables and aggregates them into tuples 
    for mean, stdev, param in zip(means, stds, params):
        print("* %f (%f) with: %r" % (mean, stdev, param))
    print("=======================================================================================")
'''    
    
#Best parameters is learning rate 0.1, n_estimator = 200, max_depth = 4
#============================================================================
#Preparing the data
print()
my_input = """
Getting started with data science and applying machine learning has never been as simple as it is now. There are many free and paid online tutorials and courses out there to help you to get started. I’ve recently started to learn, play, and work on Data Science & Machine Learning on Kaggle.com. In this brief post, I’d like to share my experience with the Kaggle Python Docker image, which simplifies the Data Scientist’s life.
Awesome #AWS monitoring introduction.
HPE Software (now @MicroFocusSW) won the platinum reader's choice #ITAWARDS 2017 in the new category #CloudMonitoring
Certified as AWS Certified Solutions Architect 
Hi, please have a look at my Udacity interview about online learning and machine learning,
Very interesting to see the  lessons learnt during the HP Operations Orchestration to CloudSlang journey. http://bit.ly/1Xo41ci 
I came across a post on devopsdigest.com and need your input: “70% DevOps organizations Unhappy with DevOps Monitoring Tools”
In a similar investigation I found out that many DevOps organizations use several monitoring tools in parallel. Senu, Nagios, LogStach and SaaS offerings such as DataDog or SignalFX to name a few. However, one element is missing: Consolidation of alerts and status in a single pane of glass, which enables fast remediation of application and infrastructure uptime and performance issues.
Sure, there are commercial tools on the market for exactly this use case but these tools are not necessarily optimized for DevOps.So, here my question to you: In your DevOps project, have you encountered that the lack of consolidation of alerts and status is a real issue? If yes, how did you approach the problem? Or is an ChatOps approach just right?
You will probably hear more and more about ChatOps - at conferences, DevOps meet-ups or simply from your co-worker at the coffee station. ChatOps is a term and concept coined by GitHub. It's about the conversation-driven development, automation, and operations.
Now the question is: why and how would I, as an ops-focused engineer, implement and use ChatOps in my organization? The next question then is: How to include my tools into the chat conversation?
Let’s begin by having a look at a use case. The Closed Looped Incidents Process (CLIP) can be rejuvenated with ChatOps. The work from the incident detection runs through monitoring until the resolution of issues in your application or infrastructure can be accelerated with improved, cross-team communication and collaboration.
In this blog post, I am going to describe and share my experience with deploying HP Operations Manager i 10.0 (OMi) on HP Helion Public Cloud. An Infrastructure as a Service platform such as HP Helion Public Cloud Compute is a great place to quickly spin-up a Linux server and install HP Operations Manager i for various use scenarios. An example of a good use case is monitoring workloads across public clouds such as AWS and Azure.
"""
#the type is just a dummy so that can be reused for the data preparing function
mydata = pd.DataFrame(data={'type': ['INFJ'], 'posts': [my_input]})

my_input, dummyType = pre_process_data(mydata, remove_stop_words=True)

my_X_cnt = cntizer.transform(my_input)
my_X_tfidf = tfizer.transform(my_X_cnt).toarray()

#Fitting and predicting the type of indicators
print()
result = []

for l in range(len(mbti_type)):
    print(mbti_type[l])
    Y = list_personality[:,l]
    
#Splitting the data into train and test sets
    seed = 7
    textSize = 0.33
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = textSize, random_state = seed)
    
#Fitting the model on the training data
    model = XGBClassifier(n_estimators = 200,
                          max_depth = 4,
                          nthread = 8,
                          learning_rate = 0.1,
                          colsample_bytree = 0.4,
                          subsample = 0.8,
                          reg_alpha = 0.5)    
    model.fit(X_train, y_train)
    
    y_predict = model.predict(my_X_tfidf)
    result.append(y_predict[0])
    
    print("%s prediction: %s" % (mbti_type[l], y_predict))
    print()

print("The result is: ", mbti_binarize(result))

