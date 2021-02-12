import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

reviews_original = pd.read_csv('C:/Users/Kelly/Documents/GitHub/DataScienceProjects/WineProject/WineReviews.csv')
reviews_original.rename(columns={'Unnamed: 0': 'WineReviewId'}, inplace = True)

# Start with examining null values
def ExamineNullValues(df):
    global null_ratio
    df_isna = df.isna()
    null_ratio = pd.DataFrame()
    null_ratio['CountNullValues'] = df_isna.sum()
    null_ratio.reset_index(inplace=True)
    null_ratio['TotalValues'] = len(df)
    null_ratio['PercentNullValues'] = null_ratio['CountNullValues']/null_ratio['TotalValues']
    null_ratio.sort_values(by=['PercentNullValues'], inplace=True)
    return null_ratio;

ExamineNullValues(reviews_original)
      
# Attribute with most null values (61% null) is 'region_2'
# Since we have enough data in 'region_1' it makes sense to remove the attribute 'region_2'
# as we don't want to impute 61% of the data or leave it as 'unknown'
reviews = reviews_original.drop(['region_2'], axis = 1)

# Attribute with next most null values (29% null) is 'designation'
# Makes sense not to delete the null values, as it seems to be valuable and accounts
# for a large percentage of records
reviews['designation'].nunique() # ~38,000 distinct values
designation = pd.DataFrame(reviews.groupby(['designation'])['WineReviewId'].agg('count'))
designation['Prevalence'] = designation['WineReviewId']/designation.sum()[0]
designation.sort_values(['Prevalence'], ascending = False, inplace = True)

# Making decision not to impute missing values with the mode since the mode is only seen
# 2% of the time. Will fill all nulls with 'undefined'
reviews.fillna(value = {'designation': 'Undefined'}, inplace = True)

# Rerun ExamineNullValues(reviews) to confirm nulls in designation have been taken care of
# and move onto the next attribute, 'taster_twitter_handle'
ExamineNullValues(reviews)

# Noting that 'taster_twitter_handle' could be representative of 'taster_name'..
# Figure out relationship between twitter handle and name
name_to_twitter = reviews.groupby(['taster_name'])['taster_twitter_handle'].agg('nunique')
twitter_to_name = reviews.groupby(['taster_twitter_handle'])['taster_name'].agg('nunique')

# Each name has only one twitter handle, but one twitter handle has two names
# The twitter handle with two names does not have any rows where the twitter handle
# is populated and the name is not, so the twitter handle is not useful here.
twitter_check = reviews[(reviews['taster_twitter_handle'] == '@worldwineguys') & \
                        (reviews['taster_name'].isnull())]

# Since the twitter handle appears to be redundant, and has so many nulls, we'll remove that column
# But first we'll check and see if there are rows where twitter handle is populated and name is not
twitter_name_count = reviews.loc[reviews['taster_twitter_handle'].notnull() & \
                                 reviews['taster_name'].isnull()]
print('Total rows with twitter handle but no name: {}'.format(len(twitter_name_count)))

# Data looks fine, looks like we can remove twitter handle once we use it to fill in any missing names
#reviews['taster_name'].fillna(value=reviews['taster_twitter_handle'], inplace = True)

# Appear to be no rows where that is the case, so we'll drop taster twitter handle.
reviews.drop(['taster_twitter_handle'], axis = 1, inplace = True)
reviews['taster_name'].fillna('Undefined',inplace = True)

# Noticing that the 'variety' column has only one record with a value of null, we'll delete that row
reviews.dropna(subset=['variety'],inplace=True)

# We have 63 rows where country and/or province is null, accounting for 0.05%
#missing_loc = reviews.loc[reviews['country'].isnull() | reviews['province'].isnull()]

# Perhaps we can fill them in based on winery
array = ['country','province']
for i in range(0,len(array)):
    col = array[i]
    wine_loc = reviews.groupby(['winery'])[col] \
                          .value_counts(normalize=True) \
                          .unstack() \
                          .fillna(0) \
                          .reset_index()
    wine_loc = wine_loc.melt(id_vars = 'winery', var_name = col, value_name = 'percent')
    wine_loc = wine_loc.sort_values('percent', ascending = False).groupby('winery', as_index=False).first()
    wine_loc.drop(wine_loc[wine_loc.percent <= 0.5].index, inplace=True)
    
    # fill na values of 'country' based on winery
    mapping = dict(wine_loc.drop(['percent'], axis = 1).values)
    reviews[col].fillna(reviews['winery'].map(mapping), inplace = True)

reviews.fillna(value = {'province': 'Undefined', 'country': 'Undefined'}, inplace = True)

ExamineNullValues(reviews)

# examine rows that have missing region_1 - do they have winery, country?
missing_reg = reviews.loc[reviews['region_1'].isnull()]
missing_reg = missing_reg.isna().sum()

# yes they do! we'll use winery, country, and province to impute missing region_1 values
reg_impute = reviews[['country','province','winery','region_1']].dropna(subset=['region_1'])
reg_impute_cts = reg_impute.groupby(['country','province','winery'])['region_1'] \
                      .value_counts(normalize=True) \
                      .unstack() \
                      .fillna(0) \
                      .reset_index()
reg_impute_cts = reg_impute_cts.melt(id_vars = ['winery','country','province'], var_name = 'region_1', value_name = 'percent')
reg_impute_cts = reg_impute_cts.sort_values('percent', ascending = False).groupby(['winery','country','province'], as_index=False).first() 
reg_impute_cts['key'] = reg_impute_cts['winery'] + '|' + reg_impute_cts['country'] + '|' + reg_impute_cts['province']

mapping = dict(reg_impute_cts[['key','region_1']].values)
reviews['key'] = reviews['winery'] + '|' + reviews['country'] + '|' + reviews['province']
reviews['region_1'].fillna(reviews['key'].map(mapping), inplace = True)
reviews.drop(['key'], axis = 1, inplace = True)

# Fill in remaining null region_1 values with 'undefined'
reviews['region_1'].fillna('Undefined', inplace = True)
ExamineNullValues(reviews)

# Only remaining attribute with null values is price
# We'll build a model to predict price and fill those values in
#-------------------------------------------------------------------

# It looks like points and price have a decent correlation
reviews[['points','price']].corr(method = 'pearson')

price_impute = reviews[['points','price']].dropna()
X = price_impute['points']
X = np.array(X.values.tolist()).reshape(-1,1)
Y = price_impute['price']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2, random_state = 0)
mod = LinearRegression()
mod.fit(X_train,Y_train)
Y_pred = mod.predict(X_test)

# Visualize the Y_test and Y_pred results
Y_test_arr = np.array(Y_test)
Y_pred_arr = np.array(Y_pred)
Y_test_arr_usingMean = np.array([Y.mean()]*len(Y_test))
dataset = pd.DataFrame({'Y_test': Y_test, 'Y_pred': list(Y_pred), 'Y_test_usingMean': list(Y_test_arr_usingMean)}, columns=['Y_test', 'Y_pred', 'Y_test_usingMean'])
plt.scatter(dataset['Y_test'], dataset['Y_pred'], color='blue',linewidth=3)
plt.scatter(dataset['Y_test'], dataset['Y_test_usingMean'], color='red',linewidth=3)

# Remove outliers to improve chart readability
dataset_mod = dataset.loc[dataset['Y_test'] <= 600]
plt.scatter(dataset_mod['Y_test'], dataset_mod['Y_pred'], color='blue',linewidth=3)
plt.scatter(dataset_mod['Y_test'], dataset_mod['Y_test_usingMean'], color='red',linewidth=3)

# Looks better, let's try one more version for all wines under $100 and see how it performs
dataset_mod = dataset.loc[dataset['Y_test'] <= 100]
plt.scatter(dataset_mod['Y_test'], dataset_mod['Y_pred'], color='blue',linewidth=3)
plt.scatter(dataset_mod['Y_test'], dataset_mod['Y_test_usingMean'], color='red',linewidth=3)

# Compute metrics to evaluate model using our predicted value
MSE = mean_squared_error(Y_test,Y_pred)
RMSE = math.sqrt(mean_squared_error(Y_test,Y_pred))
r2 = r2_score(Y_test,Y_pred)

# Compute metrics to evaluate model using simple imputed mean instead of predicted value
MSE_usingMean = mean_squared_error(Y_test,Y_test_arr_usingMean)
RMSE_usingMean = math.sqrt(mean_squared_error(Y_test,Y_test_arr_usingMean))
r2_usingMean = r2_score(Y_test,Y_test_arr_usingMean)

# Based on these metrics and our scatterplot, we conclude the linear regression model
# is not great, but is a better way to impute the missing price values than using the mean price
output = pd.DataFrame(data = [[MSE,RMSE,r2],[MSE_usingMean,RMSE_usingMean,r2_usingMean]], \
                      columns = ['MeanSquareError','RootMeanSquareError','R-Squared'], \
                      index = ['UsingPrediction','UsingMean'] \
                     )

X = reviews['points']
X = np.array(X.values.tolist()).reshape(-1,1)
Y_pred = mod.predict(X)
reviews['pred_price'] = Y_pred
reviews['price'].fillna(reviews['pred_price'], inplace = True)
reviews.drop(['pred_price'], axis = 1, inplace = True)

# Confirm no more null values
reviews.isna().sum()