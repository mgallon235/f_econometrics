import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
# Models Libraries
from scipy import stats
from scipy.stats import norm

# Variables:
'''natural logarithm of their weeklyearnings (lwklywge).'''

'''Using a density histogram or kernel density estimation (your choice) summarize
the behavior of variable weekly pay (wage) included in the sample. Find the
sample mean, sample median, and coefficient of skewness for wage. Given your
graph, are the values of the sample mean and sample median as expected? And
the coefficient of skewness? Comment.'''

# Import  data
path = os.getcwd()
df = pd.read_csv(path+"/Assig1.csv")

# Review Dataset
df.head()

# Calculate weekly averages removing log transformation
#%%
df['w_wages'] = np.exp(df['lwklywge'])

############################################# EDA
#%%
## Plot both density plots for weekly averages with and without log transformation
fig, axes = plt.subplots(1,3)

for i, el in enumerate(list(df.columns.values)):
    a= df.hist(el,ax=axes.flatten()[i],bins=30, density=True, alpha=0.6)

fig.set_size_inches(15,5)
plt.tight_layout()
plt.show()


#%%
## Create a summary statistics
def sample_dict(dataset):
    statistics = {}

    for i in dataset.columns:
        size = np.size(dataset[i])
        datatype = dataset[i].dtypes
        unique_values = dataset[i].unique().size
        mean = np.mean(dataset[i])
        stdv = np.std(dataset[i])
        min = dataset[i].min()
        per25 = dataset[i].quantile(0.25)
        median = dataset[i].quantile(0.50)
        per75 = dataset[i].quantile(0.75)
        max = dataset[i].max()
        IQRs = dataset[i].quantile(0.75) - dataset[i].quantile(0.25)
        lower_bound = (dataset[i].quantile(0.25)) - 1.5*(dataset[i].quantile(0.75) - dataset[i].quantile(0.25))
        upper_bound = (dataset[i].quantile(0.75)) + 1.5*(dataset[i].quantile(0.75) - dataset[i].quantile(0.25))

        statistics[i] = (size,datatype,unique_values,mean,stdv,min,per25,median,per75,max,IQRs,lower_bound,upper_bound)
    results = pd.DataFrame.from_dict(statistics,orient='index',columns=['size','datatype','unique_values','mean','stdv','min','per25',
                                                                        'median','per75','max','IQRs','lower_bound','upper_bound'])
    results['lower_bound'] = np.where((results['lower_bound']<0) & (results['min']>=0),0,results['lower_bound'])
    results =  round(results,2)
    results = results.to_dict('index')
    return results

#%%
# Generate summry
df_diction = sample_dict(df)
print(pd.DataFrame(df_diction))

#%%
# Calculate the number of outliers for each column
def outlier_summary(dataset,dictionary,lower,upper):
    outli = {}
    for i in dataset.columns:
        n = np.size(dataset[i])
        lower_b = round(dictionary[i][lower],3)
        outliers_lower_n = dataset[i][dataset[i] < dictionary[i][lower]].count()
        outliers_lower_perc = (dataset[i][dataset[i] < dictionary[i][lower]].count()) / np.size(dataset[i])
        upper_b = round(dictionary[i][upper],3)
        outliers_upper_n = dataset[i][dataset[i] > dictionary[i][upper]].count()
        outliers_upper_perc = (dataset[i][dataset[i] > dictionary[i][upper]].count()) / np.size(dataset[i])
        total_outliers = outliers_lower_n + outliers_upper_n
        total_outliers_perc = outliers_lower_perc + outliers_upper_perc

        outli[i] = (n,lower_b,outliers_lower_n,outliers_lower_perc,upper_b,outliers_upper_n,outliers_upper_perc,total_outliers,total_outliers_perc)
    results = pd.DataFrame.from_dict(outli,orient='index',columns=['n','lower_b','outliers_lower_n','outliers_lower_perc','upper_b','outliers_upper_n',
                                                                   'outliers_upper_perc','total_outliers','total_outliers_perc'])
    return round(results,2)

#%%
outlier_summary(df,df_diction,'lower_bound','upper_bound').T

#%%
## Summarize behaviour using: mean, median and skewness coefficients
# Calculate 4 moments of a distribution
def moment_summary(dataset):
    statistics = {}

    for i in dataset.columns:
        mean= np.mean(dataset[i])
        median = dataset[i].quantile(0.50)
        stdv = np.std(dataset[i])
        skewness = 3*(mean - median) / stdv
        mu4 = np.mean((dataset[i] - mean)**4)
        mu2 = np.mean((dataset[i] - mean)**2)
        kurtosis = mu4/(mu2**2)
        Excess_kurtosis = kurtosis - 3

        statistics[i] = (mean,median,stdv,skewness,kurtosis,Excess_kurtosis)
    results = pd.DataFrame.from_dict(statistics,orient='index',columns=['mean','median','stdv','skewness','kurtosis','Excess_kurtosis'])
    results = round(results,2)
    #Measuring Skewness
    results['is_skewed'] = results['skewness']
    results['is_skewed'] = np.where((results['skewness']> -0.5) & (results['skewness']< 0), "moderate negative skewness", results['is_skewed'])
    results['is_skewed'] = np.where((results['skewness']<= -0.5), "highly negative skewness", results['is_skewed'])
    results['is_skewed'] = np.where((results['skewness']> 0) & (results['skewness']< 0.5), "moderate positive skewness", results['is_skewed'])
    results['is_skewed'] = np.where((results['skewness']>= 0.5) , "high positive skewness", results['is_skewed'])
    results['is_skewed'] = np.where((results['skewness']== 0) , "symmetric", results['is_skewed'])

    #Measuring Kurtosis
    results['is_kurtosis'] = results['kurtosis']
    results['is_kurtosis'] = np.where((results['kurtosis']== 3) & (results['Excess_kurtosis']== 0), "mesokurtic", results['is_kurtosis'])
    results['is_kurtosis'] = np.where((results['kurtosis']< 3) & (results['Excess_kurtosis']< 0), "platykurtic", results['is_kurtosis'])
    results['is_kurtosis'] = np.where((results['kurtosis']> 3) & (results['Excess_kurtosis']> 0), "leptokurtic", results['is_kurtosis'])

    #Kurtosis description
    results['description'] = results['is_kurtosis']
    results['description'] = np.where((results['is_kurtosis']== "mesokurtic") , "extreme events are close to zero", results['description'])
    results['description'] = np.where((results['is_kurtosis']=="platykurtic") , "few values in the tails, flatter peak", results['description'])
    results['description'] = np.where((results['is_kurtosis']=="leptokurtic") , "more peaked than a normal distribution with longer tails", results['description'])

    return results

#%%
# Print results:
moment_summary(df[['lwklywge','w_wages']]).T

################################################################

## Homework
#%%
# Question 1. C
'''Provide an estimate of average weekly pay for people with 12 years of education.
This statistic would be an estimate of which parameter?'''

# define new  dataset for people with 12 years of education
df_12_edu  = df[df['educ']==12]

# Create dictionary with mean and median for new dataset
def dict_12(dataset):
    df_dict_12  = {}
    for i in  dataset.columns:
        mean = round(dataset[i].mean(),2)
        median = round(dataset[i].quantile(0.50),2)
        df_dict_12[i]  = [mean,median]
    return df_dict_12

new_dic = dict_12(df_12_edu[['w_wages','lwklywge']])


#  Print Density plots with mean and median (not showing w_wages because of outliers)

fig, axes = plt.subplots(nrows=1,ncols=2,figsize=(15,5))
axes = axes.ravel()
cols = df_12_edu[['w_wages','lwklywge']].columns

for col, ax in zip(cols,axes):
    df_12_edu[['w_wages','lwklywge']][col].plot(kind='hist',density=True, ax=ax, label=col, title=col, bins=30)
    ax.axvline(x=new_dic[col][0],color = 'red', ls='--', label = 'Mean')
    ax.axvline(x=new_dic[col][1],color = 'black', ls='--', label = 'Median')
    ax.legend()
    ax.set_ylabel('Density')

fig.tight_layout()
plt.show()

#%%
# Plot only w_wages customized
plt.hist(df_12_edu['w_wages'], bins=200, density=True, alpha=0.6, color='b',label='w_wages')
plt.set_title('Density Histogram')
plt.set_xlabel('Value')
plt.set_ylabel('Density')
plt.xlim(0,2000)
plt.axvline(x=new_dic['w_wages'][0],color = 'red', ls='--', label = 'Mean')
plt.axvline(x=new_dic['w_wages'][1],color = 'black', ls='--', label = 'Median')
plt.legend()
plt.show()


#%%
## Print Results
df[['w_wages','lwklywge']][df['educ']==12].agg(['mean','median'])
print(round(df_12,2))




#%%
'''(f) What does the thick black function you got allow you to say about the rela-
tionship between education and earnings? Explain in just a sentence. Try to

be as rigorous as possible. Can we use it to draw conclusions on the effect of
education on wages)?'''


# Group by 'age' and calculate the median for each group
# Grouping by 'educ' and calculating the mean ln(wage) for each group
conditional = df.groupby('educ')['lwklywge'].mean().reset_index()
conditional.columns = ['educ', 'avg_ln_wage']

#Calculate correlation
conditional.corr()

# Scatter plot
plt.figure(figsize=(10, 6))

# Plotting Average Salary
plt.scatter(conditional['educ'], conditional['avg_ln_wage'], marker='o', color='blue', alpha=0.7, label='Mean Data', s=70)

# Linear regression for Average Salary
slope_avg, intercept_avg, r_value_avg, _, _ = stats.linregress(conditional['educ'], conditional['avg_ln_wage'])
regression_line_avg = slope_avg * conditional['educ'] + intercept_avg
plt.plot(conditional['educ'], regression_line_avg, color='red', linestyle='--', label='Mean Fit')

# Add labels and title
plt.xlabel('educ')
plt.ylabel('avg_ln_wage')
plt.title('Scatter Plot of Education years vs. ln(Average weekly average)')

# Show the plot
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

print("Slope (m):", slope_avg)
print("Intercept (b):", intercept_avg)
print("R-squared:", r_value_avg**2)
