import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# Import dataset
df = pd.read_csv('C:/Users/Kelly/Documents/PythonProjects/Data/BankruptcyPrediction.csv')

# Check datatypes of dataframe
df.dtypes.value_counts()

# All data is numerical. Check for missing values
print('Total Null Values in Dataset: ',df.isna().sum().sum())

# Investigate count of classes
df['Bankrupt?'].value_counts()

def ModelPerformanceMetrics(Y_test,Y_pred):
    cf = confusion_matrix(Y_test,Y_pred)
    precision = precision_score(Y_test,Y_pred)
    recall = recall_score(Y_test,Y_pred)
    accuracy = accuracy_score(Y_test,Y_pred)
    fscore = f1_score(Y_test,Y_pred)
    print(cf)
    print('Precision: {} \nRecall: {} \nAccuracy: {} \nFScore: {}' \
          .format(round(precision,2),round(recall,2),round(accuracy,2),round(fscore,2)))


# First build a logistic regression model
X = df.drop(['Bankrupt?'], axis = 1)
Y = df['Bankrupt?']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

Classifier = LogisticRegression(max_iter = 1000)
Classifier = Classifier.fit(X_train,Y_train)
Y_pred = Classifier.predict(X_test)

ModelPerformanceMetrics(Y_test,Y_pred)

# Try Over-Sampling
training_set = pd.concat([X_train,Y_train], axis = 1)
df_bankrupt = training_set.loc[df['Bankrupt?'] == 1]
df_solvent = training_set.loc[df['Bankrupt?'] == 0]
multiplier = len(df_solvent)//len(df_bankrupt)
df_bankrupt_boosted = pd.concat([df_bankrupt]*multiplier, ignore_index = True)
df_oversampled = pd.concat([df_bankrupt_boosted,df_solvent], ignore_index = True)

# Separate independent and dependent variables
X_train = df_oversampled.drop(['Bankrupt?'], axis = 1)
Y_train = df_oversampled['Bankrupt?']
Classifier = LogisticRegression(max_iter = 1000)
Classifier = Classifier.fit(X_train,Y_train)
Y_pred = Classifier.predict(X_test)

ModelPerformanceMetrics(Y_test,Y_pred)

# Try Under-Sampling
df_bankrupt = df.loc[df['Bankrupt?'] == 1]
df_solvent = df.loc[df['Bankrupt?'] == 0].iloc[0:220,:]
df_undersampled = pd.concat([df_bankrupt,df_solvent], ignore_index = True)

X = df_undersampled.drop(['Bankrupt?'], axis = 1)
Y = df_undersampled['Bankrupt?']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

Classifier = LogisticRegression(max_iter = 1000)
Classifier = Classifier.fit(X_train,Y_train)
Y_pred = Classifier.predict(X_test)

ModelPerformanceMetrics(Y_test,Y_pred)

# Try Random Forest to see if it can handle both classes
X = df.drop(['Bankrupt?'], axis = 1)
Y = df['Bankrupt?']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

Classifier = RandomForestClassifier(n_estimators = 1000)
Classifier = Classifier.fit(X_train,Y_train)
Y_pred = Classifier.predict(X_test)

ModelPerformanceMetrics(Y_test,Y_pred)

# Try SMOTE
X = df.drop(['Bankrupt?'], axis = 1)
Y = df['Bankrupt?']
sm = SMOTE(sampling_strategy = 'auto', k_neighbors = 5, random_state = 0)
X_smote, Y_smote = sm.fit_resample(X, Y)
X_train, X_test, Y_train, Y_test = train_test_split(X_smote, Y_smote, test_size = 0.2, random_state = 0)

Classifier = LogisticRegression(max_iter = 1000)
Classifier = Classifier.fit(X_train,Y_train)
Y_pred = Classifier.predict(X_test)

ModelPerformanceMetrics(Y_test,Y_pred)


# Try XGBoost
X = df.drop(['Bankrupt?'], axis = 1)
Y = df['Bankrupt?']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
Classifier = XGBClassifier()
Classifier = Classifier.fit(X_train, Y_train)
Y_pred = Classifier.predict(X_test)

ModelPerformanceMetrics(Y_test,Y_pred)


# Try XGBoost with SMOTE
X = df.drop(['Bankrupt?'], axis = 1)
Y = df['Bankrupt?']
sm = SMOTE(sampling_strategy = 'auto', k_neighbors = 5, random_state = 0)
X_smote, Y_smote = sm.fit_resample(X, Y)
X_train, X_test, Y_train, Y_test = train_test_split(X_smote, Y_smote, test_size = 0.2, random_state = 0)

Classifier = XGBClassifier()
Classifier = Classifier.fit(X_train, Y_train)
Y_pred = Classifier.predict(X_test)

ModelPerformanceMetrics(Y_test,Y_pred)