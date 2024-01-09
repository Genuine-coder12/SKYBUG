import matplotlib.pyplot as plt
import re
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
import nltk
from nltk.corpus import stopwords
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

nltk.download('stopwords')
nltk.download('punkt')

movie_data_train=pd.read_csv('train_data.txt',sep=':::',names=['Title','Genre','Description'],engine='python')
movie_data_test=pd.read_csv('test_data.txt',sep=':::',names=['Title','Genre','Description'],engine='python')
# print(movie_data_train)
# print(movie_data_test)
# print(movie_data_train.shape)

# EDA of Data
genre_counts = {}
for genre in movie_data_train['Genre']:
    if genre in genre_counts:
        genre_counts[genre] += 1
    else:
        genre_counts[genre] = 1

genres = list(genre_counts.keys())
counts = list(genre_counts.values())

plt.figure(figsize=(10, 6))
plt.bar(genres, counts, color='skyblue')
plt.xlabel('Genres')
plt.ylabel('Count')
plt.title('Count of Movies per Genre')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()

# plt.show()

#Preprocessing of data
movie_data_train.dropna(inplace=True)
movie_data_test.dropna(inplace=True)

stop_words = set(stopwords.words('english'))

def Data_cleaning(text):
    # Tokenize text into words
    words = word_tokenize(text)

    # Remove stopwords
    filtered_words = [word for word in words if word.lower() not in stop_words]

    # Join filtered words back into a sentence
    filtered_summary = ' '.join(filtered_words)
    filtered_summary = re.sub(r"\s+", " ", filtered_summary).strip()
    return filtered_summary

movie_data_train["TextCleaning"] = movie_data_train["Description"].apply(Data_cleaning)
movie_data_test["TextCleaning"] = movie_data_test["Description"].apply(Data_cleaning)
# print(movie_data_train['TextCleaning'])

count_vector=CountVectorizer()
x_train_counts=count_vector.fit_transform(movie_data_train['TextCleaning'])

tfid_transformer=TfidfTransformer()
x_train_tfid=tfid_transformer.fit_transform(x_train_counts)
# print(x_train_tfid)
label_encoder=LabelEncoder()
x=x_train_tfid
y=movie_data_train['Genre']
encoded_y=label_encoder.fit_transform(y)
# print(encoded_y)

feature_train,feature_test,target_train,target_test=train_test_split(x,encoded_y,test_size=0.3)

# Naive Bayes algorithm
model1=MultinomialNB()
model1.fit(feature_train,target_train)
model_score1=model1.score(feature_train,target_train)
print(model_score1)
model_predicted1=model1.predict(feature_test)
# print(model_predicted1)
accuracy=accuracy_score(target_test,model_predicted1)
print("Naive Bayes Accuracy:", accuracy)
print(confusion_matrix(target_test,predicted))

# SVM algorithm
model2=SVC(kernel='linear', C=1)
param_grid={'C':[0.1,1,5,10,15],
            'gamma':[1,0.1,0.01],
            'kernel':['rbf','poly','sigmoid']}

grid=GridSearchCV(model2,param_grid,refit=True)
grid.fit(feature_train,target_train)
print(grid.best_estimator_)
grid_predicted2=grid.predict(feature_test)

print(grid_predicted2)
accuracy2=accuracy_score(target_test,grid_predicted2)
print("SVM Accuracy:", accuracy2)

# Logistic Regression
model3=LogisticRegression(max_iter=1000)
model3_predicted=cross_validate(model3,feature_train,target_train,cv=5)

model3.fit(feature_train,target_train)
prediction=model3.predict(feature_test)
print('Logistic Regression:',accuracy_score(target_test,prediction))
