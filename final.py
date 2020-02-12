#importing required libraries
import emoji
import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk import  pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer 
from nltk.corpus import stopwords
from math import sqrt
from math import pi
from math import exp

#Functions to calculate features
def count_url(string):
    url = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', string)
    return len(url)

def count_word(s):
    res = len(s.split())
    return res

def count_spam_words(s):
   f=open("spam-words.txt","r")
   w=f.read()
   words=s.split()
   count=0
   for i in words :
       if i in w :
           count=count+1
   return count

def count_char(s):
    count=0;
    for c in s:
        if c.isspace()!=True :
            count=count+ 1
    return count

def cal_body_richness(s):
    char_length=count_char(s)
    word_count=count_word(s)
    return char_length/word_count

def count_emoji(s):
  count=0;
  for c in s:
      if c in emoji.UNICODE_EMOJI :
          count=count+1
  return count

def count_phone_no(s):
    count=0;
    for c in s.split():
        k=len(c)
       
        if c.isdigit()&(k==11):
            count=count+1;
    return count

def count_distinct_words(s):
    thisset={""}
    for c in s.split():
        thisset.add(c)
    return len(thisset)

def count_money_tokens(s):
    l=["$","cash","Rs"]
    cnt=0;
    for c in s.split():
        if c in l:
            cnt=cnt+1
    return cnt   

#def count_non_english_words(s):
#    from nltk.corpus import words
#    word_list=words.words()
#    res=0
#    for w in s.split():
#        if w not in word_list:
#            res=res+1
#    return res

def count_exression_words(s):
   f=open("expressions_words.txt","r")
   w=f.read()
   words=s.split()
   count=0
   for i in words :
       if i in w :
           count=count+1
   return count    

#reading the dataset
df=pd.read_csv("SpamDetectordataset.csv")

#normalisation
normalised_msg=list()
for i in range(len(df)) : 
   input_str=df.loc[i,"Message"]
   #removing proper noun   
   s=pos_tag(word_tokenize(input_str))
   ans=list()
   for i in s:
       if(i[1] != "NNP"):
           ans.append(i[0])
   removed_propernoun=' '.join(ans)   
   #converting to lower case
   lower_s=removed_propernoun.lower()   
   #removing punctuations
   punctuations = '''!()-[]{};:'"\,<>./?@#%^&*_~'''
   no_punct = ""
   for char in lower_s:
       if char not in punctuations:
           no_punct = no_punct + char
   #tokenizing the text
   tokens=word_tokenize(no_punct)
   #expanding the abbreviations
   expan = open('expansions.txt','r').read()
   contractions=eval(expan)
   expanding_str=list()      
   for y in tokens :
       if y in contractions :
           expanding_str.append(contractions[y])
       else :
           expanding_str.append(y)
   joinnedstr=''.join(expanding_str)
   expanded_str=word_tokenize(no_punct)
   #lemmatizating
   lemmatizer = WordNetLemmatizer()
   lemmatized_msg=list()
   for word in expanded_str:
      lemmatized_msg.append(lemmatizer.lemmatize(word))
   #stemming
   stemmer= PorterStemmer()
   stemmed_msg=list()
   for word in lemmatized_msg:
       stemmed_msg.append(stemmer.stem(word))
   #removing stop words
   stop_words = set(stopwords.words('english')) 
   final_msg = list()
   for w in stemmed_msg: 
       if w not in stop_words: 
           final_msg.append(w)
   temp=' '.join(final_msg)
   normalised_msg.append(temp)

#preparation of feature matrix   
feature_vector=list()   
for i in range(len(df)) : 
  s=df.loc[i,"Message"]
  temp=list()
  k=count_word(s)
  temp.append(k)
  k=count_char(s)
  temp.append(k)
  k=cal_body_richness(s)
  temp.append(k)
  k=count_emoji(s)
  temp.append(k)
  k=count_url(s)
  temp.append(k)
  k=count_phone_no(s)
  temp.append(k)
  s=normalised_msg[i]
  k=count_distinct_words(s);
  temp.append(k)
  k=count_money_tokens(s);
  temp.append(k)
  k=count_spam_words(s)
  temp.append(k)
  #k=count_non_english_words(s)
  #temp.append(k)
  k=count_exression_words(s)
  temp.append(k)
  feature_vector.append(temp)

#preparation of final dataset  
X=pd.DataFrame(feature_vector,columns=['Word_Count','Character_Count','Body_Richness','Emoji_Count','URL_Count','PhoneNo_Count','DistinctWords_Count','MoneyToken_Count','SpamWords_Count','ExpressionWords_count'])
Y=list()
for i in range(len(df)) : 
  Y.append(df.loc[i, "Output"]) 

final_y=list()
for i in Y:
    if i=="ham" :
        final_y.append(0)
    else:
        final_y.append(1)
        
result = np.hstack((np.array(X), np.atleast_2d(np.array(final_y)).T))

#Naive Bayes as feature implementation
# Splitting the dataset based on class value
def split(dset):
	split = dict()
	for i in range(len(dset)):
		feature_vector = dset[i]
		val_cls = feature_vector[-1]
		if (val_cls not in split):
			split[val_cls] = list()
		split[val_cls].append(feature_vector)
	return split

# Calculating the mean
def mean_numbers(nums):
	return sum(nums)/float(len(nums))

# Calculating the standard deviation
def stdev_numbers(nums):
	avg_nums = mean_numbers(nums)
	vari = sum([(x-avg_nums)**2 for x in nums]) / float(len(nums)-1)
	return sqrt(vari)

# Calculating the mean,standard deviation and count for eah column
def summarize(dset):
	smrize = [(mean_numbers(column), stdev_numbers(column), len(column)) for column in zip(*dset)]
	del(smrize[-1])
	return smrize

# Calculating statistics for each row after spliting by class
def summarize_by_class(dset):
	separated = split(dset)
	summaries = dict()
	for val_cls, rows in separated.items():
		summaries[val_cls] = summarize(rows)
	return summaries

# Calculating the Gaussian probability distribution function for x
def cal_probability(x, mean, stdev):
    expo = exp(-((x-mean)**2 / (2 * stdev**2 )))
    try:
        return (1 / (sqrt(2 * pi) * stdev)) * expo
    except:
        return 1

# Calculating the probabilities of predicting each class for a given row
def calculate_class_probabilities(summaries, row):
	total_rows = sum([summaries[label][0][2] for label in summaries])
	probabilities = dict()
	for class_value, class_summaries in summaries.items():
		probabilities[class_value] = summaries[class_value][0][2]/float(total_rows)
		for i in range(len(class_summaries)):
			mean, stdev, _ = class_summaries[i]
			probabilities[class_value] *= cal_probability(row[i], mean, stdev)
	return probabilities

#Calculating class probabilities
temp0=list()
temp1=list()
summaries = summarize_by_class(result)
for i in range(len(result)) :
    probability = calculate_class_probabilities(summaries, result[i])
    temp0.append(probability[0])
    temp1.append(probability[1])

#inserting the calculated class probability 
final_dataset=pd.DataFrame(result)    
final_dataset.insert(10, 'ham_probability', temp0)  
final_dataset.insert(11, 'spam_probability', temp1)   
final_dataset.columns = ['Word_Count','Character_Count','Body_Richness','Emoji_Count','URL_Count','PhoneNo_Count','DistinctWords_Count','MoneyToken_Count','SpamWords_Count','ExpressionWords_count','ham_probability','spam_probaility','result']   

#Final feature matrix and response vector
final_X=final_dataset.iloc[:,:-1].values
final_y=final_dataset.iloc[:,-1].values

#splitting the dataset into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(final_X,final_y, test_size = 0.2, random_state=0)

#Applying various Classification Algorithms
#Applying Logistic Regression
from sklearn.linear_model import LogisticRegression
classifierLR = LogisticRegression(random_state = 0)
classifierLR.fit(X_train,y_train)
y_predLR=classifierLR.predict(X_test)

#Calculating the accuracy, precision and recall for logistic regression by checking it against the test set
from sklearn.metrics import confusion_matrix
cmLR = confusion_matrix(y_test,y_predLR)
from sklearn import metrics
accuracyLR=metrics.accuracy_score(y_test,y_predLR)
precisionLR=cmLR[0][0]/(cmLR[0][0]+cmLR[1][0])
recallLR=cmLR[0][0]/(cmLR[0][0]+cmLR[0][1])
#printing the values
print("Accuracy of Logistic Regression algorithm : ",accuracyLR)
#print("Confusion Matrix for Logistic Regression: ",cmLR)
print("Precision for Logistic Regression: ",precisionLR) 
print("Recall for Logistic Regression: ",recallLR)

#Applying K Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier
classifierKNN = KNeighborsClassifier(n_neighbors = 13, metric = 'minkowski' , p=2)
classifierKNN.fit(X_train,y_train)
y_predKNN=classifierKNN.predict(X_test)

#Calculating the accuracy, precision and recall for K Nearest Neighbors by checking it against the test set
cmKNN = confusion_matrix(y_test,y_predKNN)
accuracyKNN=metrics.accuracy_score(y_test,y_predKNN)
precisionKNN=cmKNN[0][0]/(cmKNN[0][0]+cmKNN[1][0])
recallKNN=cmKNN[0][0]/(cmKNN[0][0]+cmKNN[0][1])
#printing the values
print("Accuracy of K Nearest Neighbors algorithm : ",accuracyKNN)
#print("Confusion Matrix for K Nearest Neighbors: ",cmLR)
print("Precision for K Nearest Neighbor: ",precisionKNN)
print("Recall for K Nearest Neighbor: ",recallKNN) 

#Applying Support Vector Machine
from sklearn.svm import SVC
classifierSVM = SVC(kernel='rbf',random_state=0)
classifierSVM.fit(X_train,y_train)
y_predSVM=classifierSVM.predict(X_test)

#Calculating the accuracy, precision and recall for Support Vector Machine by checking it against the test set
cmSVM = confusion_matrix(y_test,y_predSVM)
accuracySVM=metrics.accuracy_score(y_test,y_predSVM)
precisionSVM=cmSVM[0][0]/(cmSVM[0][0]+cmSVM[1][0])
recallSVM=cmSVM[0][0]/(cmSVM[0][0]+cmSVM[0][1])
#printing the values
print("Accuracy of Support Vector Machine algorithm : ",accuracySVM)
#print("Confusion Matrix for Support Vector Machine: ",cmLR)
print("Precision for Support Vector Machine: ",precisionSVM)
print("Recall for Support Vector Machine: ",recallSVM) 

#Applying Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
classifierDT = DecisionTreeClassifier(criterion='entropy',random_state=0)
classifierDT.fit(X_train,y_train)
y_predDT=classifierDT.predict(X_test)

#Calculating the accuracy, precision and recall for Decision Tree Classifier by checking it against the test set
cmDT = confusion_matrix(y_test,y_predDT)
accuracyDT=metrics.accuracy_score(y_test,y_predDT)
precisionDT=cmDT[0][0]/(cmDT[0][0]+cmDT[1][0])
recallDT=cmDT[0][0]/(cmDT[0][0]+cmDT[0][1])
#printing the values
print("Accuracy of Decision Tree algorithm : ",accuracyDT)
#print("Confusion Matrix for Decision Tree: ",cmLR)
print("Precision for Decision Tree: ",precisionDT) 
print("Recall for Decision Tree: ",recallDT)

#Applying Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
classifierRFC = RandomForestClassifier(n_estimators = 50,criterion='entropy',random_state=0)
classifierRFC.fit(X_train,y_train)
y_predRFC=classifierRFC.predict(X_test)
#Calculating the accuracy, precision and recall for Random Forest Classifier by checking it against the test set
cmRFC = confusion_matrix(y_test,y_predRFC)
accuracyRFC=metrics.accuracy_score(y_test,y_predRFC)
precisionRFC=cmRFC[0][0]/(cmRFC[0][0]+cmRFC[1][0])
recallRFC=cmRFC[0][0]/(cmRFC[0][0]+cmRFC[0][1])
#printing the values
print("Accuracy of Random Forest Classification algorithm : ",accuracyRFC)
#print("Confusion Matrix for Random Forest Classification: ",cmLR)
print("Precision for Random Forest Classification: ",precisionRFC)
print("Recall for Random Forest Classification: ",recallRFC) 
 
#Appying Bagging(Bootstrapping and agregation) algorithm
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
n_Kfold= model_selection.KFold(n_splits=10,random_state=7)
base_class=DecisionTreeClassifier()
classifierBC= BaggingClassifier(base_estimator=base_class,n_estimators=100, random_state=7)
y_predBC=model_selection.cross_val_score(classifierBC,final_X,final_y,cv=n_Kfold)
print(mean_numbers(y_predBC))

