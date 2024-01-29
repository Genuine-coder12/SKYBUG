import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score,confusion_matrix,roc_curve,classification_report
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB

data=pd.read_csv('Churn_Modelling.csv')
# print(data.info())

# print(data.isnull().sum())
encoder=LabelEncoder()

data["Gender"]=encoder.fit_transform(data["Gender"])
data["Geography"]=encoder.fit_transform(data["Geography"])
# print(data.head())

# EDA
count_churn = data["Exited"].value_counts()
count_prod = data["NumOfProducts"].value_counts()

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.bar(count_prod.index, count_prod.values)
plt.xlabel("No. of Product")
plt.ylabel("Count")
plt.title("Number of Products")

plt.subplot(2, 1, 2)
plt.pie(count_churn, labels=["Not_Exited", "Exited"])
plt.title("Exited")

plt.tight_layout()

plt.figure(figsize=(12, 6))
plt.hist(data['Age'], bins=30, edgecolor='black')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title("Age histogram")

plt.show()


# Preprocessing
data.drop(['CustomerId','Surname'],axis=1,inplace=True)
data.dropna(axis=1)

# Model training
x=data.drop(['Exited'],axis=1)
y=data['Exited']

feature_train,feature_test,target_train,target_test=train_test_split(
    x,y,test_size=0.3
)

model1=RandomForestClassifier(
    n_estimators=5,
    max_depth=5,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)
model1.fit(feature_train,target_train)
model1_pred=model1.predict(feature_test)
print("Accuracy of Random_Forest_Classifier:",accuracy_score(target_test,model1_pred))

# Naive Bayes
scale=StandardScaler()
feature_train_scaled=scale.fit_transform(feature_train)
feature_test_scaled=scale.fit_transform(feature_test)

model2=GaussianNB()
model2.fit(feature_train_scaled,target_train)
model2_predict=model2.predict(feature_test_scaled)

print("Accuracy of Naive_Bayes_Classifier:",accuracy_score(target_test,model2_predict))
# print("\nClassification Report Of Naive Bayes:\n", classification_report(target_test, model2_predict))

# Decision Tree
dt_classifier = DecisionTreeClassifier()
param_grid = {
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Use GridSearchCV to search for the best hyperparameters
grid_search = GridSearchCV(dt_classifier, param_grid, cv=5, scoring='accuracy')
grid_search.fit(feature_train_scaled, target_train)

best_params = grid_search.best_params_
best_dt_classifier = DecisionTreeClassifier(**best_params)

best_dt_classifier.fit(feature_train_scaled, target_train)
dt_predict = best_dt_classifier.predict(feature_test_scaled)

print("Best Hyperparameters:", best_params)
print("Accuracy of Decision_Tree_Classifier:", accuracy_score(target_test, dt_predict))

# Logistic Regression
logistic_classifier = LogisticRegression()

param_grid_logistic = {
    'C': [0.001, 0.01, 0.1, 1, 10]
}

grid_search_logistic = GridSearchCV(logistic_classifier, param_grid_logistic, cv=5, scoring='accuracy')
grid_search_logistic.fit(feature_train_scaled, target_train)

best_params_logistic = grid_search_logistic.best_params_
best_logistic_classifier = LogisticRegression(**best_params_logistic)
best_logistic_classifier.fit(feature_train_scaled, target_train)

logistic_predict = best_logistic_classifier.predict(feature_test_scaled)

print("Best Hyperparameters for Logistic Regression:", best_params_logistic)
print("Accuracy of Logistic Regression:", accuracy_score(target_test, logistic_predict))


# Gradient Boosting
gradient_classifier = GradientBoostingClassifier()

param_grid_gradient = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.001, 0.01, 0.1],
    'max_depth': [4, 5],
}

grid_search_gradient = GridSearchCV(gradient_classifier, param_grid_gradient, cv=5, scoring='accuracy')
grid_search_gradient.fit(feature_train_scaled, target_train)

best_params_gradient = grid_search_gradient.best_params_
best_gradient_classifier = GradientBoostingClassifier(**best_params_gradient)
best_gradient_classifier.fit(feature_train_scaled, target_train)

gradient_predict = best_gradient_classifier.predict(feature_test_scaled)

print("Best Hyperparameters for Gradient Boosting:", best_params_gradient)
print("Accuracy of Gradient Boosting:", accuracy_score(target_test, gradient_predict))
# print("\nClassification Report Of Gradient Boosting:\n", classification_report(target_test, gradient_predict))
