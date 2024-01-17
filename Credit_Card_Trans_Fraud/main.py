# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

# Load the training and test datasets
data_train = pd.read_csv('fraudTrain.csv')
data_test = pd.read_csv('fraudTest.csv')

# Display information about the training data
data_train.info()

# Display column names in the training data
data_train.columns

# Preprocessing

# Drop columns with high cardinality and unnecessary information
data_train.drop(columns=['Unnamed: 0', 'cc_num', 'first', 'last', 'street', 'city', 'state',
                         'dob', 'trans_date_trans_time', 'trans_num'], inplace=True)

# Drop rows with missing values
data_train.dropna(ignore_index=True)

# Drop columns for the test data as well
data_test.drop(columns=['Unnamed: 0', 'cc_num', 'first', 'last', 'street', 'city', 'state',
                        'dob', 'trans_date_trans_time', 'trans_num'], inplace=True)
data_test.dropna(ignore_index=True)

# Use LabelEncoder for categorical features

# Training Data
data_train['category'] = LabelEncoder().fit_transform(data_train['category'])
data_train['gender'] = LabelEncoder().fit_transform(data_train['gender'])
data_train['merchant'] = LabelEncoder().fit_transform(data_train['merchant'])
data_train['job'] = LabelEncoder().fit_transform(data_train['job'])

# Test Data
data_test['category'] = LabelEncoder().fit_transform(data_test['category'])
data_test['gender'] = LabelEncoder().fit_transform(data_test['gender'])
data_test['merchant'] = LabelEncoder().fit_transform(data_test['merchant'])
data_test['job'] = LabelEncoder().fit_transform(data_test['job'])

# Exploratory Data Analysis (EDA)

# Plot histograms for the target variable "is_fraud" in both training and test data
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(data_train["is_fraud"], bins=2, color='skyblue', edgecolor='black')
plt.xlabel("is_fraud")
plt.ylabel("Count_train")
plt.title("is_fraud Histogram")
plt.xticks([0.25, 0.75], ["No", "YES"])
plt.tight_layout()

plt.subplot(1, 2, 2)
plt.hist(data_test["is_fraud"], bins=2, color='skyblue', edgecolor='black')
plt.xlabel("is_fraud")
plt.ylabel("Count_test")
plt.title("is_fraud Histogram")
plt.xticks([0.25, 0.75], ["No", "YES"])
plt.tight_layout()
plt.show()

# Model Training

# Standardize features using StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(data_train.drop(columns=["is_fraud"], inplace=False))
y_train = data_train['is_fraud']
x_test = scaler.fit_transform(data_test.drop(columns=["is_fraud"], inplace=False))
y_test = data_test['is_fraud']

# Logistic Regression
logistic_regression_params = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
logistic_regression_model = LogisticRegression()
logistic_regression_grid = GridSearchCV(logistic_regression_model, logistic_regression_params, cv=5)
logistic_regression_grid.fit(x_train, y_train)
logistic_regression_best = logistic_regression_grid.best_estimator_
logistic_regression_predict = logistic_regression_best.predict(x_test)
logistic_regression_acc = accuracy_score(y_test, logistic_regression_predict)
print("Logistic Regression model accuracy:", logistic_regression_acc)
print("Best Parameters:", logistic_regression_grid.best_params_)
print("Confusion Matrix:\n", confusion_matrix(y_test, logistic_regression_predict))

# Support Vector Machine (SVM)
svm_params = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
svm_model = SVC()
svm_grid = GridSearchCV(svm_model, svm_params, cv=5)
svm_grid.fit(x_train, y_train)
svm_best = svm_grid.best_estimator_
svm_predict = svm_best.predict(x_test)
print("SVM model accuracy:", accuracy_score(y_test, svm_predict))
print("Best Parameters:", svm_grid.best_params_)
print("Confusion Matrix:\n", confusion_matrix(y_test, svm_predict))

# Random Forest Classifier
random_forest_params = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20, 30]}
random_forest_model = RandomForestClassifier()
random_forest_grid = GridSearchCV(random_forest_model, random_forest_params, cv=5)
random_forest_grid.fit(x_train, y_train)
random_forest_best = random_forest_grid.best_estimator_
random_forest_predict = random_forest_best.predict(x_test)
print("Random Forest Classifier model accuracy:", accuracy_score(y_test, random_forest_predict))
print("Best Parameters:", random_forest_grid.best_params_)
print("Confusion Matrix:\n", confusion_matrix(y_test, random_forest_predict))

# Decision Tree Classifier
decision_tree_params = {'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10]}
decision_tree_model = DecisionTreeClassifier()
decision_tree_grid = GridSearchCV(decision_tree_model, decision_tree_params, cv=5)
decision_tree_grid.fit(x_train, y_train)
decision_tree_best = decision_tree_grid.best_estimator_
decision_tree_predict = decision_tree_best.predict(x_test)
print("Decision Tree Classifier accuracy:", accuracy_score(y_test, decision_tree_predict))
print("Best Parameters:", decision_tree_grid.best_params_)
print("Confusion Matrix:\n", confusion_matrix(y_test, decision_tree_predict))
