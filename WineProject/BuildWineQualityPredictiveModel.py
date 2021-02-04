import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sklearn.metrics as metrics
from sklearn.metrics import roc_curve, confusion_matrix, f1_score, recall_score, precision_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

# Import and read csv file
quality_path = 'C:/Users/Kelly/Documents/GitHub/DataScienceProjects/WineProject/WineQuality.csv'
quality = pd.read_csv(quality_path)

# Get an idea of variable correlation
correlation = quality.iloc[:,1:].corr()

# Check distribution of quality values to determine split for 'good' and 'bad' wine
plt.hist(quality['Quality'])
top_50_perc = np.percentile(quality['Quality'],50)

# Create supervisor based on 'quality' rating
quality['IsConsumerPreferredWine'] = np.where(quality['Quality'] >= 6,1,0)

# Confirm no null values in columns and supervisor
quality.isnull().sum()

# Separate data into independent and dependent variables
X = quality.iloc[:,1:-2] # X should include all attributes except 'Id', 'Quality', and supervisor
Y = quality.iloc[:,-1] # Y should be supervisor, determined from above step based on 'Quality'


def BuildModel(X,Y,Algorithm):
        
    # Split data into train and test set
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
    
    # Save training set if need to reference in future when evaluating model
    TrainingSet = pd.merge(X_train, pd.DataFrame(Y_train), left_index  = True, right_index = True)
    
    # Build model and create prediction for Y_test
    if Algorithm == 'LogisticRegression':
        Classifier = LogisticRegression()

    if Algorithm == 'DecisionTree':
        Classifier = tree.DecisionTreeClassifier()

    if Algorithm == 'RandomForest':
        Classifier = RandomForestClassifier(n_estimators = 1000)
    
    if Algorithm == 'NaiveBayes':
        Classifier = GaussianNB()
        
    # Train model and predict values for X_test dataset    
    Classifier = Classifier.fit(X_train,Y_train)
    Y_pred = Classifier.predict(X_test)
    
    # Evaluate model with confusion matrix
    cf = confusion_matrix(Y_test,Y_pred)
    f_score = f1_score(Y_test, Y_pred)
    precision = precision_score(Y_test,Y_pred)
    recall = precision_score(Y_test,Y_pred)
    accuracy = accuracy_score(Y_test,Y_pred)

    print('Using the {} classifier, we receive an accuracy score of {} and f score of {}.' \
          .format(Algorithm,round(accuracy,4),round(f_score,4)))

# Easy way to loop through and build all three models to compare metrics
def BuildAndCompareModels(array):
    for i in range(0,len(array)):
        BuildModel(X,Y,array[i])