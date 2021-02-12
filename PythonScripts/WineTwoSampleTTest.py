import pandas as pd
import numpy as np
import scipy
from scipy.stats import ttest_ind, ttest_ind_from_stats, shapiro, mannwhitneyu
from scipy.special import stdtr

# Import and read csv file
quality_path = 'C:/Users/Kelly/Documents/GitHub/DataScienceProjects/WineProject/WineQuality.csv'
quality = pd.read_csv(quality_path)

def PerformTest(indVariable):
    # Get 50th percentile
    percentile = np.percentile(quality[indVariable], 50)
    
    # Analyze relationship between independent variable and wine quality
    a = quality.loc[quality[indVariable] < percentile]
    b = quality.loc[quality[indVariable] > percentile]
    
    # Check for normality 
    a_norm = shapiro(a[indVariable])
    b_norm = shapiro(b[indVariable])
    
    if (a_norm[1] < 0.05 or b_norm[1] < 0.05):
        print('This is not a normal distribution. We cannot use a 2-sample T-test, so we will use a Mann-Whitney U Test.')
        
        # Perform Mann-Whitney U Test
        s, p = mannwhitneyu(a['Quality'], b['Quality'], alternative = 'two-sided')
        print(p)
        
    else:
        # Compute mean for each sample
        a_mean = a['Quality'].mean()
        b_mean = b['Quality'].mean()
        
        # Compute variance for each sample to see if they are approximately equal
        a_var = np.var(a['Quality'])
        b_var = np.var(b['Quality'])
        acidic_ratio = a_var/b_var
        
        if acidic_ratio <= 4: # adjust this...
            # Variances are confirmed approximately equal, perform 2-sample t-test
            s, p = ttest_ind(a['Quality'], b['Quality'], equal_var=True)
            
            print('Wines with {} above {} had a mean Quality rating of {}.'.format(indVariable,percentile,round(b_mean,2)))
            print('Wines with {} below {} had a mean Quality rating of {}.'.format(indVariable,percentile,round(a_mean,2)))
            
    if(p >= 0.05):
        print('Fail to reject null hypothesis that wine quality is not significantly different between wines with {} below and above {} with a p-value of {}.'.format(indVariable,percentile,p))
    elif(p < 0.05):
        print('Reject the null hypothesis that wine quality is not significantly different between wines with {} below and above {} with a p-value of {}.'.format(indVariable,percentile,p))
