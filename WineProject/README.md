# DataScienceProjects
# Red Wine Quality Data

# Python code written to build and evaluate three different machine learning algorithms

# Line 18 - 23
# Take 'Quality' value from dataset to build supervisor. Based on approximately top 50% of quality ratings
# being above 6, set all Quality values greater than 6 to 1 for supervisor - 'IsConsumerPreferredWine' - 
# and all values less than or equal to 6 to 0, indicating they are not preferred by consumers.

# Line 33
# Use BuildModel function to build either a Logistic Regression, Decision Tree, or Random Forest classifier

# Line 65
# Use BuildAndCompareModels to input an array (Ex. [LogisticRegression, DecisionTree]) and train all models
# in that array to compare their output.
