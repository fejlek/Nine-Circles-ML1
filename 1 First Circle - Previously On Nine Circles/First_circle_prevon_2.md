# The First Circle: Previously On Nine Circles, Part Two 
## Logistic regression

<br/>
Jiří Fejlek

2025-12-05
<br/>

We continue our summary of regression methods with logistic regression. We will use the dataset obtained from <https://www.kaggle.com/datasets/dileep070/heart-disease-prediction-using-logistic-regression/data> based on the Framingham Heart Study <https://www.framinghamheartstudy.org>. The Framingham Heart Study is a long-term, ongoing cardiovascular cohort study of residents of Framingham, Massachusetts, that began in 1948 to identify factors contributing to cardiovascular diseases, including the effects of high blood pressure and smoking.

## Table of Contents

- [Cardiovascular Study on Residents of the Town of Framingham, Massachusetts](#framingham)
- [Complete Case Analysis](#complete-case)
    - [Model Fit and Examining Effects of Predictors](#fit-inference)
    - [Model Diagnostics](#diagnostics)
    - [Measures of Predictive Performance](#performance)
        - [Prediction Accuracy (percentage of correct predictions)](#accuracy)
        - [Specificity & Sensitivity, ROC curve](#roc)
        - [Strictly Proper Scoring Rules (logarithmic score and Brier score)](#scoring)
        - [Calibration](#calibration)
        - [Decision Curve Analysis](#dca)
    - [Model Validation](#validation)
- [Mean and Most-Frequent Imputation](#mean-imputation)
- [k-NN Imputation](#knn)
- [MissForrest Imputation](#missforrest)
- [Regression Imputation (via chained equation)](#mice)
- [References](#references)


## Cardiovascular Study on Residents of the Town of Framingham, Massachusetts <a class="anchor" id="framingham"></a>

The Framingham Heart Study contains the following information about 4238 individuals. Each individual was examined and then followed for 10 years for the outcome of developing coronary heart disease.

* **Sex** 
* **Age** - Age (at the time of examination)
* **Education** - Four levels: no high school, high school, college, and college graduate
* **Current Smoker** - Whether or not the subject was a  smoker (at the time of examination)
* **Cigs Per Day** - Cigarettes smoked on average in one day
* **BP Meds** - Whether or not the subject was on blood pressure medication 
* **Prevalent Stroke** - Whether or not the subject had previously had a stroke
* **Prevalent Hyp** - Whether or not the subject was hypertensive
* **Diabetes** - Whether or not the subject had diabetes
* **Tot Chol** -  Total cholesterol level
* **Sys BP** - Systolic blood pressure 
* **Dia BP** -  Diastolic blood pressure
* **BMI** - Body Mass Index
* **Glucose** - Glucose level 
* **TenYearCHD** -  Whether or not a coronary heart disease occurred in 10 years after examination


First, let us load the dataset.


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.model_selection import KFold

np.set_printoptions(legacy='1.25')
```


```python
framingham = pd.read_csv('C:/Users/elini/Desktop/nine circles 2/framingham.csv')
framingham
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>male</th>
      <th>age</th>
      <th>education</th>
      <th>currentSmoker</th>
      <th>cigsPerDay</th>
      <th>BPMeds</th>
      <th>prevalentStroke</th>
      <th>prevalentHyp</th>
      <th>diabetes</th>
      <th>totChol</th>
      <th>sysBP</th>
      <th>diaBP</th>
      <th>BMI</th>
      <th>heartRate</th>
      <th>glucose</th>
      <th>TenYearCHD</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>39</td>
      <td>4.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>195.0</td>
      <td>106.0</td>
      <td>70.0</td>
      <td>26.97</td>
      <td>80.0</td>
      <td>77.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>46</td>
      <td>2.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>250.0</td>
      <td>121.0</td>
      <td>81.0</td>
      <td>28.73</td>
      <td>95.0</td>
      <td>76.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>48</td>
      <td>1.0</td>
      <td>1</td>
      <td>20.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>245.0</td>
      <td>127.5</td>
      <td>80.0</td>
      <td>25.34</td>
      <td>75.0</td>
      <td>70.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>61</td>
      <td>3.0</td>
      <td>1</td>
      <td>30.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>225.0</td>
      <td>150.0</td>
      <td>95.0</td>
      <td>28.58</td>
      <td>65.0</td>
      <td>103.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>46</td>
      <td>3.0</td>
      <td>1</td>
      <td>23.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>285.0</td>
      <td>130.0</td>
      <td>84.0</td>
      <td>23.10</td>
      <td>85.0</td>
      <td>85.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4233</th>
      <td>1</td>
      <td>50</td>
      <td>1.0</td>
      <td>1</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>313.0</td>
      <td>179.0</td>
      <td>92.0</td>
      <td>25.97</td>
      <td>66.0</td>
      <td>86.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4234</th>
      <td>1</td>
      <td>51</td>
      <td>3.0</td>
      <td>1</td>
      <td>43.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>207.0</td>
      <td>126.5</td>
      <td>80.0</td>
      <td>19.71</td>
      <td>65.0</td>
      <td>68.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4235</th>
      <td>0</td>
      <td>48</td>
      <td>2.0</td>
      <td>1</td>
      <td>20.0</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>248.0</td>
      <td>131.0</td>
      <td>72.0</td>
      <td>22.00</td>
      <td>84.0</td>
      <td>86.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4236</th>
      <td>0</td>
      <td>44</td>
      <td>1.0</td>
      <td>1</td>
      <td>15.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>210.0</td>
      <td>126.5</td>
      <td>87.0</td>
      <td>19.16</td>
      <td>86.0</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4237</th>
      <td>0</td>
      <td>52</td>
      <td>2.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>269.0</td>
      <td>133.5</td>
      <td>83.0</td>
      <td>21.47</td>
      <td>80.0</td>
      <td>107.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>4238 rows × 16 columns</p>
</div>




```python
framingham.dtypes
```




    male                 int64
    age                  int64
    education          float64
    currentSmoker        int64
    cigsPerDay         float64
    BPMeds             float64
    prevalentStroke      int64
    prevalentHyp         int64
    diabetes             int64
    totChol            float64
    sysBP              float64
    diaBP              float64
    BMI                float64
    heartRate          float64
    glucose            float64
    TenYearCHD           int64
    dtype: object



We shorten the names of the predictors.


```python
framingham = framingham.rename(columns={'male': 'Sex', 'age': 'Age', 'currentSmoker': 'Smoker', \
                                        'prevalentStroke': 'Stroke', 'prevalentHyp': 'Hyp', \
                                        'diabetes': 'Diab', 'TenYearCHD': 'TCHD', 'sysBP': 'SysP', \
                                        'diaBP': 'DiaP', 'heartRate': 'Hrate', 'cigsPerDay': 'Cig', \
                                        'totChol': 'Chol', 'BPMeds': 'Meds', 'education': 'Edu', \
                                        'glucose': 'Gluc'})
```

Next, we check whether any data is missing.


```python
framingham.duplicated(keep = False).any(axis = None)
```




    False




```python
framingham.isna().any(axis = None)
```




    True



<br/> Some values are indeed missing.


```python
framingham.isna().any(axis = 1).sum()
```




    582




```python
framingham.isna().sum(axis = 0)
```




    male                 0
    age                  0
    education          105
    currentSmoker        0
    cigsPerDay          29
    BPMeds              53
    prevalentStroke      0
    prevalentHyp         0
    diabetes             0
    totChol             50
    sysBP                0
    diaBP                0
    BMI                 19
    heartRate            1
    glucose            388
    TenYearCHD           0
    dtype: int64



The fraction of rows with some missing values is 582/4238 ~ 0.14. This is a significantly greater value than 3%, which is a rule-of-thumb value for which it should not matter much how the observation with missing values is treated [[1](#1)]. Thus, we need to take some care with our analysis as far as the missing data is concerned.

Let us check the predictors.


```python
framingham['Sex'].value_counts() # 0: female, 1:male
```




    Sex
    0    2419
    1    1819
    Name: count, dtype: int64




```python
framingham['Edu'].value_counts()
```




    Edu
    1.0    1720
    2.0    1253
    3.0     687
    4.0     473
    Name: count, dtype: int64




```python
framingham['Smoker'].value_counts()
```




    Smoker
    0    2144
    1    2094
    Name: count, dtype: int64




```python
framingham['Meds'].value_counts()
```




    Meds
    0.0    4061
    1.0     124
    Name: count, dtype: int64




```python
framingham['Stroke'].value_counts()
```




    Stroke
    0    4213
    1      25
    Name: count, dtype: int64




```python
framingham['Hyp'].value_counts()
```




    Hyp
    0    2922
    1    1316
    Name: count, dtype: int64




```python
framingham['Diab'].value_counts()
```




    Diab
    0    4129
    1     109
    Name: count, dtype: int64




```python
fig, axs = plt.subplots(1,3)
axs[0].hist(framingham['Age'], bins=20)
axs[0].set_xlabel('Age')
axs[0].set_ylabel('Frequency')
axs[1].hist(framingham['SysP'], bins=20)  
axs[1].set_xlabel('SysP')    
axs[2].hist(framingham['DiaP'], bins=20)  
axs[2].set_xlabel('DiaP');    
```




    Text(0.5, 0, 'DiaP')




    
![png](First_circle_prevon_2_files/output_20_1.png)
    



```python
fig, axs = plt.subplots(1,4)
axs[0].hist(framingham['Hrate'], bins=20)
axs[0].set_xlabel('Hrate')
axs[0].set_ylabel('Frequency')
axs[1].hist(framingham['Cig'], bins=20)  
axs[1].set_xlabel('Cig')    
axs[2].hist(framingham['Chol'], bins=20)  
axs[2].set_xlabel('Chol')
axs[3].hist(framingham['Gluc'], bins=20)  
axs[3].set_xlabel('Gluc');
```


    
![png](First_circle_prevon_2_files/output_21_0.png)
    


Overall, the values and their distributions seem reasonable. Some minima and maxima are pretty extreme, but none of these seem impossible to occur. Several factors, namely **BP Meds**, **Diabetes**, and especially  **Prevalent Stroke**, have a low number of cases, which could hurt the accuracy of estimates of their effect. Still, these predictors seem too important to be just ignored.   

We will remove **Current Smoker** from our model and keep just **Cigs Per Day** since no smoker reports that he/she smoke zero cigarettes per day on average. Thus, we opt to quantify the effect of smoking in our model using a more informative numerical predictor **Cigs Per Day.** Otherwise, we will consider all predictors for modeling.


```python
((framingham['Smoker'] == 1) & (framingham['Cig'] == 0)).any()
```




    False



## Complete Case Analysis <a class="anchor" id="complete-case"></a>

Before we proceed to model with the imputation of missing values, we will perform *complete case analysis* (listwise deletion) for future comparison with other approaches. We should remember that the inference based on complete case analysis is valid under the missing completely at random (MCAR) condition (the probability of missing is the same for all cases), i.e., complete case analysis under MCAR produces unbiased regression estimates [[2](#2)]. If this is not the case (missingness depends on the data or on some unobserved variables), then these estimates may be severely biased. Another disadvantage of complete-case analysis is that it can be wasteful with the data. On the other hand, a complete case analysis is very simple to perform.


```python
framingham_complete = framingham.loc[~framingham.isna().any(axis = 1)].reset_index(drop=True)
```

Before we fit the data, let us check whether some variables are not too correlated.


```python
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF
ind = [*range(3),*range(4,15)]
VIF_dataframe = framingham_complete.iloc[:,ind].assign(const=1)

VIF_vals = [VIF(VIF_dataframe, i) 
        for i in range(0, VIF_dataframe.shape[1])]
VIF_vals = pd.DataFrame({'VIF':VIF_vals},index=VIF_dataframe.columns)
VIF_vals
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>VIF</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Sex</th>
      <td>1.201334</td>
    </tr>
    <tr>
      <th>Age</th>
      <td>1.353394</td>
    </tr>
    <tr>
      <th>Edu</th>
      <td>1.055094</td>
    </tr>
    <tr>
      <th>Cig</th>
      <td>1.201720</td>
    </tr>
    <tr>
      <th>Meds</th>
      <td>1.111183</td>
    </tr>
    <tr>
      <th>Stroke</th>
      <td>1.017398</td>
    </tr>
    <tr>
      <th>Hyp</th>
      <td>2.050781</td>
    </tr>
    <tr>
      <th>Diab</th>
      <td>1.615851</td>
    </tr>
    <tr>
      <th>Chol</th>
      <td>1.116276</td>
    </tr>
    <tr>
      <th>SysP</th>
      <td>3.766397</td>
    </tr>
    <tr>
      <th>DiaP</th>
      <td>2.995337</td>
    </tr>
    <tr>
      <th>BMI</th>
      <td>1.226777</td>
    </tr>
    <tr>
      <th>Hrate</th>
      <td>1.093220</td>
    </tr>
    <tr>
      <th>Gluc</th>
      <td>1.637347</td>
    </tr>
    <tr>
      <th>const</th>
      <td>190.519374</td>
    </tr>
  </tbody>
</table>
</div>



### Model Fit and Examining Effects of Predictors <a class="anchor" id="fit-inference"></a>

Since we are dealing with the binary response, we will be fitting a logistic regression model $\mathrm{log} \frac{p}{1-p} = X\beta$, where $p$ is the probability of the response 1 and $X$ is the model matrix of our predictors and $\beta$ are the parameters. When selecting our model matrix, we should consider the *effective sample size*. For binary response, the effective sample size is the number of observations for the less represented response; in our case, it is 557 (even though we technically have 3656 observations). Thus, our data reasonably support approximately 557/10 ~ 56 to 557/20 ~ 28 parameters [[1](#1)].


```python
(framingham['TCHD'] == 0).sum()
```




    3594




```python
(framingham['TCHD'] == 1).sum()
```




    644



Since we have only 14 predictors, we can include some nonlinearities and interactions in the model.  We will consider cubic splines with 3 DOF for all numerical predictors. We will also consider linear interactions of age and sex with risk factors (i.e., the whole model has 53 parameters).


```python
lr_full = smf.logit(formula='TCHD ~ Sex + bs(Age, df=3) + C(Edu, Poly) + bs(Cig, df=3)  + Meds + Stroke + Hyp + Diab + bs(Chol, df=3) + \
                     bs(SysP, df=3) + bs(DiaP, df=3) + bs(BMI, df=3) + bs(Hrate, df=3) + bs(Gluc, df=3) + \
                     Age:(Cig + Stroke + Hyp + Diab + Chol + SysP + DiaP + BMI + Hrate + Gluc) + \
                     Sex:(Cig + Stroke + Hyp + Diab + Chol + SysP + DiaP + BMI + Hrate + Gluc)', data=framingham_complete)
lr_full_fit = lr_full.fit(disp=0)                              
print(lr_full_fit.summary())
```

                               Logit Regression Results                           
    ==============================================================================
    Dep. Variable:                   TCHD   No. Observations:                 3656
    Model:                          Logit   Df Residuals:                     3603
    Method:                           MLE   Df Model:                           52
    Date:                Wed, 03 Dec 2025   Pseudo R-squ.:                  0.1342
    Time:                        18:14:07   Log-Likelihood:                -1350.9
    converged:                       True   LL-Null:                       -1560.3
    Covariance Type:            nonrobust   LLR p-value:                 8.903e-59
    ==========================================================================================
                                 coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------------------
    Intercept                 -1.9624      1.514     -1.296      0.195      -4.929       1.004
    C(Edu, Poly).Linear       -0.0179      0.118     -0.151      0.880      -0.250       0.214
    C(Edu, Poly).Quadratic     0.1783      0.119      1.502      0.133      -0.054       0.411
    C(Edu, Poly).Cubic         0.0182      0.118      0.154      0.878      -0.214       0.250
    Sex                       -2.0806      1.292     -1.611      0.107      -4.612       0.451
    bs(Age, df=3)[0]           2.6640      1.535      1.735      0.083      -0.345       5.673
    bs(Age, df=3)[1]           3.7168      2.016      1.844      0.065      -0.234       7.668
    bs(Age, df=3)[2]           5.0271      2.956      1.701      0.089      -0.766      10.820
    bs(Cig, df=3)[0]          -0.2333      0.942     -0.248      0.804      -2.079       1.613
    bs(Cig, df=3)[1]           2.5789      1.705      1.513      0.130      -0.762       5.920
    bs(Cig, df=3)[2]          -2.6528      3.043     -0.872      0.383      -8.617       3.311
    Meds                       0.1463      0.242      0.605      0.545      -0.328       0.621
    Stroke                    -4.1172      6.309     -0.653      0.514     -16.483       8.248
    Hyp                       -0.5911      1.084     -0.545      0.585      -2.715       1.533
    Diab                       1.9127      2.891      0.662      0.508      -3.753       7.578
    bs(Chol, df=3)[0]          0.2089      2.090      0.100      0.920      -3.887       4.305
    bs(Chol, df=3)[1]          0.3803      3.638      0.105      0.917      -6.750       7.511
    bs(Chol, df=3)[2]          7.8619      5.693      1.381      0.167      -3.297      19.020
    bs(SysP, df=3)[0]          1.1031      2.540      0.434      0.664      -3.875       6.081
    bs(SysP, df=3)[1]         -3.4485      4.818     -0.716      0.474     -12.892       5.995
    bs(SysP, df=3)[2]          1.2432      7.848      0.158      0.874     -14.140      16.626
    bs(DiaP, df=3)[0]         -2.6149      2.088     -1.252      0.211      -6.708       1.478
    bs(DiaP, df=3)[1]          3.7373      3.084      1.212      0.226      -2.307       9.781
    bs(DiaP, df=3)[2]          4.2823      4.739      0.904      0.366      -5.006      13.570
    bs(BMI, df=3)[0]          -0.5357      1.673     -0.320      0.749      -3.816       2.744
    bs(BMI, df=3)[1]           1.5176      2.627      0.578      0.563      -3.630       6.665
    bs(BMI, df=3)[2]           1.1433      4.108      0.278      0.781      -6.909       9.195
    bs(Hrate, df=3)[0]         1.2152      1.719      0.707      0.480      -2.154       4.585
    bs(Hrate, df=3)[1]        -1.0632      2.112     -0.503      0.615      -5.203       3.077
    bs(Hrate, df=3)[2]         0.9434      3.573      0.264      0.792      -6.059       7.946
    bs(Gluc, df=3)[0]         -4.0686      2.522     -1.613      0.107      -9.013       0.875
    bs(Gluc, df=3)[1]         -0.3432      4.379     -0.078      0.938      -8.926       8.240
    bs(Gluc, df=3)[2]         -7.6685      7.140     -1.074      0.283     -21.663       6.326
    Age:Cig                    0.0002      0.001      0.422      0.673      -0.001       0.001
    Age:Stroke                 0.0806      0.109      0.738      0.461      -0.134       0.295
    Age:Hyp                    0.0200      0.019      1.036      0.300      -0.018       0.058
    Age:Diab                  -0.0481      0.049     -0.980      0.327      -0.144       0.048
    Age:Chol               -9.784e-05      0.000     -0.664      0.506      -0.000       0.000
    Age:SysP                   0.0004      0.001      0.687      0.492      -0.001       0.001
    Age:DiaP                  -0.0013      0.001     -1.582      0.114      -0.003       0.000
    Age:BMI                   -0.0004      0.002     -0.286      0.775      -0.004       0.003
    Age:Hrate              -3.898e-05      0.001     -0.072      0.942      -0.001       0.001
    Age:Gluc                   0.0005      0.000      1.348      0.178      -0.000       0.001
    Sex:Cig                   -0.0057      0.010     -0.565      0.572      -0.026       0.014
    Sex:Stroke                 0.4937      1.051      0.470      0.638      -1.566       2.553
    Sex:Hyp                   -0.4977      0.296     -1.684      0.092      -1.077       0.082
    Sex:Diab                   0.8637      0.707      1.222      0.222      -0.522       2.249
    Sex:Chol                   0.0044      0.002      1.846      0.065      -0.000       0.009
    Sex:SysP                   0.0099      0.008      1.223      0.221      -0.006       0.026
    Sex:DiaP                   0.0054      0.013      0.423      0.672      -0.020       0.031
    Sex:BMI                   -0.0139      0.028     -0.499      0.618      -0.069       0.041
    Sex:Hrate                  0.0059      0.009      0.665      0.506      -0.012       0.023
    Sex:Gluc                  -0.0018      0.005     -0.378      0.705      -0.011       0.007
    ==========================================================================================
    


```python
len(lr_full_fit.params)
```




    53



To test the significance of a given variable (e.g., **Cigs per day**) on the probability of developing **TCHD**, we can use a likelihood ratio test.


```python
# likelihood ratio tests
from scipy.stats.distributions import chi2   

LR_test = pd.DataFrame(index=['Sex','Age','Edu','Cig','Meds','Stroke','Hyp','Diab','Chol','SysP','DiabP','BMI','Hrate','Gluc'], \
                       columns = ['Deviance','DoF','P-value'])


lr_no_sex = smf.logit(formula='TCHD ~ bs(Age, df=3) + C(Edu, Poly) + bs(Cig, df=3) + Meds + Stroke + Hyp + Diab + bs(Chol, df=3) + \
                     bs(SysP, df=3) + bs(DiaP, df=3) + bs(BMI, df=3) + bs(Hrate, df=3) + bs(Gluc, df=3) + \
                     Age:(Cig + Stroke + Hyp + Diab + Chol + SysP + DiaP + BMI + Hrate + Gluc)', data=framingham_complete)
                     
LR_test.iloc[0,0] = 2*(lr_full_fit.llf-lr_no_sex.fit(disp=0).llf)
LR_test.iloc[0,1] = (lr_full.df_model - lr_no_sex.df_model)
LR_test.iloc[0,2] = 1 - chi2.cdf(LR_test.iloc[0,0],LR_test.iloc[0,1])
                          

lr_no_age = smf.logit(formula='TCHD ~ Sex + C(Edu, Poly) + bs(Cig, df=3) + Meds + Stroke + Hyp + Diab + bs(Chol, df=3) + \
                     bs(SysP, df=3) + bs(DiaP, df=3) + bs(BMI, df=3) + bs(Hrate, df=3) + bs(Gluc, df=3) + \
                     Sex:(Cig + Stroke + Hyp + Diab + Chol + SysP + DiaP + BMI + Hrate + Gluc)', data=framingham_complete)
LR_test.iloc[1,0] = 2*(lr_full_fit.llf-lr_no_age.fit(disp=0).llf)
LR_test.iloc[1,1] = (lr_full.df_model - lr_no_age.df_model)
LR_test.iloc[1,2] = 1 - chi2.cdf(LR_test.iloc[1,0],LR_test.iloc[1,1])


lr_no_edu = smf.logit(formula='TCHD ~ Sex + bs(Age, df=3) + bs(Cig, df=3) + Meds + Stroke + Hyp + Diab + bs(Chol, df=3) + \
                     bs(SysP, df=3) + bs(DiaP, df=3) + bs(BMI, df=3) + bs(Hrate, df=3) + bs(Gluc, df=3) + \
                     Age:(Cig + Stroke + Hyp + Diab + Chol + SysP + DiaP + BMI + Hrate + Gluc) + \
                     Sex:(Cig + Stroke + Hyp + Diab + Chol + SysP + DiaP + BMI + Hrate + Gluc)', data=framingham_complete)
LR_test.iloc[2,0] = 2*(lr_full_fit.llf-lr_no_edu.fit(disp=0).llf)
LR_test.iloc[2,1] = (lr_full.df_model - lr_no_edu.df_model)
LR_test.iloc[2,2] = 1 - chi2.cdf(LR_test.iloc[2,0],LR_test.iloc[2,1])
                
      
lr_no_cig = smf.logit(formula='TCHD ~ Sex + bs(Age, df=3) + C(Edu, Poly) + Meds + Stroke + Hyp + Diab + bs(Chol, df=3) + \
                     bs(SysP, df=3) + bs(DiaP, df=3) + bs(BMI, df=3) + bs(Hrate, df=3) + bs(Gluc, df=3) + \
                     Age:(Stroke + Hyp + Diab + Chol + SysP + DiaP + BMI + Hrate + Gluc) + \
                     Sex:(Stroke + Hyp + Diab + Chol + SysP + DiaP + BMI + Hrate + Gluc)', data=framingham_complete)                     
LR_test.iloc[3,0] = 2*(lr_full_fit.llf-lr_no_cig.fit(disp=0).llf)
LR_test.iloc[3,1] = (lr_full.df_model - lr_no_cig.df_model)
LR_test.iloc[3,2] = 1 - chi2.cdf(LR_test.iloc[3,0],LR_test.iloc[3,1])


lr_no_meds = smf.logit(formula='TCHD ~ Sex + bs(Age, df=3) + C(Edu, Poly) + bs(Cig, df=3) + Stroke + Hyp + Diab + bs(Chol, df=3) + \
                     bs(SysP, df=3) + bs(DiaP, df=3) + bs(BMI, df=3) + bs(Hrate, df=3) + bs(Gluc, df=3) + \
                     Age:(Cig + Stroke + Hyp + Diab + Chol + SysP + DiaP + BMI + Hrate + Gluc) + \
                     Sex:(Cig + Stroke + Hyp + Diab + Chol + SysP + DiaP + BMI + Hrate + Gluc)', data=framingham_complete)
LR_test.iloc[4,0] = 2*(lr_full_fit.llf-lr_no_meds.fit(disp=0).llf)
LR_test.iloc[4,1] = (lr_full.df_model - lr_no_meds.df_model)
LR_test.iloc[4,2] = 1 - chi2.cdf(LR_test.iloc[4,0],LR_test.iloc[4,1])


lr_no_stroke = smf.logit(formula='TCHD ~ Sex + bs(Age, df=3) + C(Edu, Poly) + bs(Cig, df=3) + Meds + Hyp + Diab + bs(Chol, df=3) + \
                     bs(SysP, df=3) + bs(DiaP, df=3) + bs(BMI, df=3) + bs(Hrate, df=3) + bs(Gluc, df=3) + \
                     Age:(Cig + Hyp + Diab + Chol + SysP + DiaP + BMI + Hrate + Gluc) + \
                     Sex:(Cig + Hyp + Diab + Chol + SysP + DiaP + BMI + Hrate + Gluc)', data=framingham_complete)
LR_test.iloc[5,0] = 2*(lr_full_fit.llf-lr_no_stroke.fit(disp=0).llf)
LR_test.iloc[5,1] = (lr_full.df_model - lr_no_stroke.df_model)
LR_test.iloc[5,2] = 1 - chi2.cdf(LR_test.iloc[5,0],LR_test.iloc[5,1])


lr_no_hyp = smf.logit(formula='TCHD ~ Sex + bs(Age, df=3) + C(Edu, Poly) + bs(Cig, df=3) + Meds + Stroke + Diab + bs(Chol, df=3) + \
                     bs(SysP, df=3) + bs(DiaP, df=3) + bs(BMI, df=3) + bs(Hrate, df=3) + bs(Gluc, df=3) + \
                     Age:(Cig + Stroke + Diab + Chol + SysP + DiaP + BMI + Hrate + Gluc) + \
                     Sex:(Cig + Stroke + Diab + Chol + SysP + DiaP + BMI + Hrate + Gluc)', data=framingham_complete)
LR_test.iloc[6,0] = 2*(lr_full_fit.llf-lr_no_hyp.fit(disp=0).llf)
LR_test.iloc[6,1] = (lr_full.df_model - lr_no_hyp.df_model)
LR_test.iloc[6,2] = 1 - chi2.cdf(LR_test.iloc[6,0],LR_test.iloc[6,1])


lr_no_diab = smf.logit(formula='TCHD ~ Sex + bs(Age, df=3) + C(Edu, Poly) + bs(Cig, df=3) + Meds + Stroke + Hyp + bs(Chol, df=3) + \
                     bs(SysP, df=3) + bs(DiaP, df=3) + bs(BMI, df=3) + bs(Hrate, df=3) + bs(Gluc, df=3) + \
                     Age:(Cig + Stroke + Hyp + Chol + SysP + DiaP + BMI + Hrate + Gluc) + \
                     Sex:(Cig + Stroke + Hyp + Chol + SysP + DiaP + BMI + Hrate + Gluc)', data=framingham_complete)
LR_test.iloc[7,0] = 2*(lr_full_fit.llf-lr_no_diab.fit(disp=0).llf)
LR_test.iloc[7,1] = (lr_full.df_model - lr_no_diab.df_model)
LR_test.iloc[7,2] = 1 - chi2.cdf(LR_test.iloc[7,0],LR_test.iloc[7,1])


lr_no_chol = smf.logit(formula='TCHD ~ Sex + bs(Age, df=3) + C(Edu, Poly) + bs(Cig, df=3) + Meds + Stroke + Hyp + Diab + \
                     bs(SysP, df=3) + bs(DiaP, df=3) + bs(BMI, df=3) + bs(Hrate, df=3) + bs(Gluc, df=3) + \
                     Age:(Cig + Stroke + Hyp + Diab + SysP + DiaP + BMI + Hrate + Gluc) + \
                     Sex:(Cig + Stroke + Hyp + Diab + SysP + DiaP + BMI + Hrate + Gluc)', data=framingham_complete)
LR_test.iloc[8,0] = 2*(lr_full_fit.llf-lr_no_chol.fit(disp=0).llf)
LR_test.iloc[8,1] = (lr_full.df_model - lr_no_chol.df_model)
LR_test.iloc[8,2] = 1 - chi2.cdf(LR_test.iloc[8,0],LR_test.iloc[8,1]) 


lr_no_sysp = smf.logit(formula='TCHD ~ Sex + bs(Age, df=3) + C(Edu, Poly) + bs(Cig, df=3) + Meds + Stroke + Hyp + Diab + bs(Chol, df=3) + \
                     bs(DiaP, df=3) + bs(BMI, df=3) + bs(Hrate, df=3) + bs(Gluc, df=3) + \
                     Age:(Cig + Stroke + Hyp + Diab + Chol + DiaP + BMI + Hrate + Gluc) + \
                     Sex:(Cig + Stroke + Hyp + Diab + Chol + DiaP + BMI + Hrate + Gluc)', data=framingham_complete)                   
LR_test.iloc[9,0] = 2*(lr_full_fit.llf-lr_no_sysp.fit(disp=0).llf)
LR_test.iloc[9,1] = (lr_full.df_model - lr_no_sysp.df_model)
LR_test.iloc[9,2] = 1 - chi2.cdf(LR_test.iloc[9,0],LR_test.iloc[9,1])   


lr_no_diap = smf.logit(formula='TCHD ~ Sex + bs(Age, df=3) + C(Edu, Poly) + bs(Cig, df=3) + Meds + Stroke + Hyp + Diab + bs(Chol, df=3) + \
                     bs(SysP, df=3) + bs(BMI, df=3) + bs(Hrate, df=3) + bs(Gluc, df=3) + \
                     Age:(Cig + Stroke + Hyp + Diab + Chol + SysP + BMI + Hrate + Gluc) + \
                     Sex:(Cig + Stroke + Hyp + Diab + Chol + SysP + BMI + Hrate + Gluc)', data=framingham_complete)
LR_test.iloc[10,0] = 2*(lr_full_fit.llf-lr_no_diap.fit(disp=0).llf)
LR_test.iloc[10,1] = (lr_full.df_model - lr_no_diap.df_model)
LR_test.iloc[10,2] = 1 - chi2.cdf(LR_test.iloc[10,0],LR_test.iloc[10,1])   
                  

lr_no_bmi = smf.logit(formula='TCHD ~ Sex + bs(Age, df=3) + C(Edu, Poly) + bs(Cig, df=3) + Meds + Stroke + Hyp + Diab + bs(Chol, df=3) + \
                     bs(SysP, df=3) + bs(DiaP, df=3) + bs(Hrate, df=3) + bs(Gluc, df=3) + \
                     Age:(Cig + Stroke + Hyp + Diab + Chol + SysP + DiaP + Hrate + Gluc) + \
                     Sex:(Cig + Stroke + Hyp + Diab + Chol + SysP + DiaP + Hrate + Gluc)', data=framingham_complete)                     
LR_test.iloc[11,0] = 2*(lr_full_fit.llf-lr_no_bmi.fit(disp=0).llf)
LR_test.iloc[11,1] = (lr_full.df_model - lr_no_bmi.df_model)
LR_test.iloc[11,2] = 1 - chi2.cdf(LR_test.iloc[11,0],LR_test.iloc[11,1]) 


lr_no_hrate = smf.logit(formula='TCHD ~ Sex + bs(Age, df=3) + C(Edu, Poly) + bs(Cig, df=3) + Meds + Stroke + Hyp + Diab + bs(Chol, df=3) + \
                     bs(SysP, df=3) + bs(DiaP, df=3) + bs(BMI, df=3) + bs(Gluc, df=3) + \
                     Age:(Cig + Stroke + Hyp + Diab + Chol + SysP + DiaP + BMI + Gluc) + \
                     Sex:(Cig + Stroke + Hyp + Diab + Chol + SysP + DiaP + BMI + Gluc)', data=framingham_complete)
LR_test.iloc[12,0] = 2*(lr_full_fit.llf-lr_no_hrate.fit(disp=0).llf)
LR_test.iloc[12,1] = (lr_full.df_model - lr_no_hrate.df_model)
LR_test.iloc[12,2] = 1 - chi2.cdf(LR_test.iloc[12,0],LR_test.iloc[12,1])                      


lr_no_gluc = smf.logit(formula='TCHD ~ Sex + bs(Age, df=3) + C(Edu, Poly) + bs(Cig, df=3) + Meds + Stroke + Hyp + Diab + bs(Chol, df=3) + \
                     bs(SysP, df=3) + bs(DiaP, df=3) + bs(BMI, df=3) + bs(Hrate, df=3) + \
                     Age:(Cig + Stroke + Hyp + Diab + Chol + SysP + DiaP + BMI + Hrate) + \
                     Sex:(Cig + Stroke + Hyp + Diab + Chol + SysP + DiaP + BMI + Hrate)', data=framingham_complete)
LR_test.iloc[13,0] = 2*(lr_full_fit.llf-lr_no_gluc.fit(disp=0).llf)
LR_test.iloc[13,1] = (lr_full.df_model - lr_no_gluc.df_model)
LR_test.iloc[13,2] = 1 - chi2.cdf(LR_test.iloc[13,0],LR_test.iloc[13,1])                      
                     
LR_test
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Deviance</th>
      <th>DoF</th>
      <th>P-value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Sex</th>
      <td>32.14357</td>
      <td>11.0</td>
      <td>0.000723</td>
    </tr>
    <tr>
      <th>Age</th>
      <td>106.658236</td>
      <td>13.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Edu</th>
      <td>2.941309</td>
      <td>3.0</td>
      <td>0.400763</td>
    </tr>
    <tr>
      <th>Cig</th>
      <td>24.693428</td>
      <td>5.0</td>
      <td>0.00016</td>
    </tr>
    <tr>
      <th>Meds</th>
      <td>0.36142</td>
      <td>1.0</td>
      <td>0.547719</td>
    </tr>
    <tr>
      <th>Stroke</th>
      <td>2.827239</td>
      <td>3.0</td>
      <td>0.419036</td>
    </tr>
    <tr>
      <th>Hyp</th>
      <td>6.407344</td>
      <td>3.0</td>
      <td>0.093389</td>
    </tr>
    <tr>
      <th>Diab</th>
      <td>3.713417</td>
      <td>3.0</td>
      <td>0.294119</td>
    </tr>
    <tr>
      <th>Chol</th>
      <td>15.402675</td>
      <td>5.0</td>
      <td>0.008773</td>
    </tr>
    <tr>
      <th>SysP</th>
      <td>17.607744</td>
      <td>5.0</td>
      <td>0.00348</td>
    </tr>
    <tr>
      <th>DiabP</th>
      <td>14.196162</td>
      <td>5.0</td>
      <td>0.01441</td>
    </tr>
    <tr>
      <th>BMI</th>
      <td>1.787007</td>
      <td>5.0</td>
      <td>0.877761</td>
    </tr>
    <tr>
      <th>Hrate</th>
      <td>2.466258</td>
      <td>5.0</td>
      <td>0.781568</td>
    </tr>
    <tr>
      <th>Gluc</th>
      <td>15.169986</td>
      <td>5.0</td>
      <td>0.00966</td>
    </tr>
  </tbody>
</table>
</div>



We observe that **Sex**, **Age**, **Cig**, **Chol**, **SysP**, **DiabP**, and **Gluc** are significant risk factors in the model. Let us examine the predicted marginal effects (along with their confidence intervals) based on our model. We will plot the effects for both sexes at the median value of the remaining predictors.


```python
median_values = framingham_complete.median();
```


```python
# age & sex
age_seq = np.array([*range(80,145,5)])/2

male_plot = framingham_complete.copy().iloc[range(len(age_seq)),:]
male_plot.iloc[:] = median_values
male_plot['Sex'] = 1
male_plot['Age'] = age_seq

female_plot = male_plot.copy()
female_plot['Sex'] = 0

predict_male = lr_full.fit(disp=0).get_prediction(male_plot).summary_frame(alpha=0.05)
predict_female = lr_full.fit(disp=0).get_prediction(female_plot).summary_frame(alpha=0.05)


plt.fill_between(male_plot['Age'], predict_male['ci_upper'],\
                 predict_male['ci_lower'], alpha=.25,color = 'blue')
plt.plot(male_plot['Age'],predict_male['predicted'], label='Male',color = 'blue')
plt.fill_between(female_plot['Age'], predict_female['ci_upper'],\
                 predict_female['ci_lower'], alpha=.25,color = 'red')
plt.plot(female_plot['Age'],predict_female['predicted'], label='Female',color = 'red')
plt.xlabel('Age')
plt.ylabel('Predicted Probability of TCHD')
plt.legend();
```


    
![png](First_circle_prevon_2_files/output_39_0.png)
    



```python
# cig & sex
cig_seq = np.array([*range(0,60,5)])

male_plot = framingham_complete.copy().iloc[range(len(cig_seq)),:]
male_plot.iloc[:] = median_values
male_plot['Sex'] = 1
male_plot['Cig'] = cig_seq

female_plot = male_plot.copy()
female_plot['Sex'] = 0

predict_male = lr_full.fit(disp=0).get_prediction(male_plot).summary_frame(alpha=0.05)
predict_female = lr_full.fit(disp=0).get_prediction(female_plot).summary_frame(alpha=0.05)


plt.fill_between(male_plot['Cig'], predict_male['ci_upper'],\
                 predict_male['ci_lower'], alpha=.25,color = 'blue')
plt.plot(male_plot['Cig'],predict_male['predicted'], label='Male',color = 'blue')
plt.fill_between(female_plot['Cig'], predict_female['ci_upper'],\
                 predict_female['ci_lower'], alpha=.25,color = 'red')
plt.plot(female_plot['Cig'],predict_female['predicted'], label='Female',color = 'red')
plt.xlabel('Cig')
plt.ylabel('Predicted Probability of TCHD')
plt.legend();
```


    
![png](First_circle_prevon_2_files/output_40_0.png)
    



```python
# edu & sex
edu_seq = [1,2,3,4]

male_plot = framingham_complete.copy().iloc[range(len(edu_seq)),:]
male_plot.iloc[:] = median_values
male_plot['Sex'] = 1
male_plot['Edu'] = edu_seq

female_plot = male_plot.copy()
female_plot['Sex'] = 0

predict_male = lr_full.fit(disp=0).get_prediction(male_plot).summary_frame(alpha=0.05)
predict_female = lr_full.fit(disp=0).get_prediction(female_plot).summary_frame(alpha=0.05)


plt.fill_between(male_plot['Edu'], predict_male['ci_upper'],\
                 predict_male['ci_lower'], alpha=.25,color = 'blue')
plt.plot(male_plot['Edu'],predict_male['predicted'], label='Male',color = 'blue')
plt.fill_between(female_plot['Edu'], predict_female['ci_upper'],\
                 predict_female['ci_lower'], alpha=.25,color = 'red')
plt.plot(female_plot['Edu'],predict_female['predicted'], label='Female',color = 'red')
plt.xlabel('Edu')
plt.ylabel('Predicted Probability of TCHD')
plt.legend();
```


    
![png](First_circle_prevon_2_files/output_41_0.png)
    



```python
# chol & sex
chol_seq = np.array([*range(120,600,30)])

male_plot = framingham_complete.copy().iloc[range(len(chol_seq)),:]
male_plot.iloc[:] = median_values
male_plot['Sex'] = 1
male_plot['Chol'] = chol_seq

female_plot = male_plot.copy()
female_plot['Sex'] = 0

predict_male = lr_full.fit(disp=0).get_prediction(male_plot).summary_frame(alpha=0.05)
predict_female = lr_full.fit(disp=0).get_prediction(female_plot).summary_frame(alpha=0.05)


plt.fill_between(male_plot['Chol'], predict_male['ci_upper'],\
                 predict_male['ci_lower'], alpha=.25,color = 'blue')
plt.plot(male_plot['Chol'],predict_male['predicted'], label='Male',color = 'blue')
plt.fill_between(female_plot['Chol'], predict_female['ci_upper'],\
                 predict_female['ci_lower'], alpha=.25,color = 'red')
plt.plot(female_plot['Chol'],predict_female['predicted'], label='Female',color = 'red')
plt.xlabel('Chol')
plt.ylabel('Predicted Probability of TCHD')
plt.legend();
```


    
![png](First_circle_prevon_2_files/output_42_0.png)
    



```python
# sysp & sex
sysp_seq = np.array([*range(85,295,10)])

male_plot = framingham_complete.copy().iloc[range(len(sysp_seq)),:]
male_plot.iloc[:] = median_values
male_plot['Sex'] = 1
male_plot['SysP'] = sysp_seq

female_plot = male_plot.copy()
female_plot['Sex'] = 0

predict_male = lr_full.fit(disp=0).get_prediction(male_plot).summary_frame(alpha=0.05)
predict_female = lr_full.fit(disp=0).get_prediction(female_plot).summary_frame(alpha=0.05)


plt.fill_between(male_plot['SysP'], predict_male['ci_upper'],\
                 predict_male['ci_lower'], alpha=.25,color = 'blue')
plt.plot(male_plot['SysP'],predict_male['predicted'], label='Male',color = 'blue')
plt.fill_between(female_plot['SysP'], predict_female['ci_upper'],\
                 predict_female['ci_lower'], alpha=.25,color = 'red')
plt.plot(female_plot['SysP'],predict_female['predicted'], label='Female',color = 'red')
plt.xlabel('SysP')
plt.ylabel('Predicted Probability of TCHD')
plt.legend();
```


    
![png](First_circle_prevon_2_files/output_43_0.png)
    



```python
# diap & sex
diap_seq = np.array([*range(50,140,5)])

male_plot = framingham_complete.copy().iloc[range(len(diap_seq)),:]
male_plot.iloc[:] = median_values
male_plot['Sex'] = 1
male_plot['DiaP'] = diap_seq

female_plot = male_plot.copy()
female_plot['Sex'] = 0

predict_male = lr_full.fit(disp=0).get_prediction(male_plot).summary_frame(alpha=0.05)
predict_female = lr_full.fit(disp=0).get_prediction(female_plot).summary_frame(alpha=0.05)


plt.fill_between(male_plot['DiaP'], predict_male['ci_upper'],\
                 predict_male['ci_lower'], alpha=.25,color = 'blue')
plt.plot(male_plot['DiaP'],predict_male['predicted'], label='Male',color = 'blue')
plt.fill_between(female_plot['DiaP'], predict_female['ci_upper'],\
                 predict_female['ci_lower'], alpha=.25,color = 'red')
plt.plot(female_plot['DiaP'],predict_female['predicted'], label='Female',color = 'red')
plt.xlabel('DiaP')
plt.ylabel('Predicted Probability of TCHD')
plt.legend();
```


    
![png](First_circle_prevon_2_files/output_44_0.png)
    



```python
# BMI & sex
bmi_seq = np.array([*range(20,55,5)])

male_plot = framingham_complete.copy().iloc[range(len(bmi_seq)),:]
male_plot.iloc[:] = median_values
male_plot['Sex'] = 1
male_plot['BMI'] = bmi_seq

female_plot = male_plot.copy()
female_plot['Sex'] = 0

predict_male = lr_full.fit(disp=0).get_prediction(male_plot).summary_frame(alpha=0.05)
predict_female = lr_full.fit(disp=0).get_prediction(female_plot).summary_frame(alpha=0.05)


plt.fill_between(male_plot['BMI'], predict_male['ci_upper'],\
                 predict_male['ci_lower'], alpha=.25,color = 'blue')
plt.plot(male_plot['BMI'],predict_male['predicted'], label='Male',color = 'blue')
plt.fill_between(female_plot['BMI'], predict_female['ci_upper'],\
                 predict_female['ci_lower'], alpha=.25,color = 'red')
plt.plot(female_plot['BMI'],predict_female['predicted'], label='Female',color = 'red')
plt.xlabel('BMI')
plt.ylabel('Predicted Probability of TCHD')
plt.legend();
```


    
![png](First_circle_prevon_2_files/output_45_0.png)
    



```python
# Hrate & sex
hrate_seq = np.array([*range(45,145,5)])

male_plot = framingham_complete.copy().iloc[range(len(hrate_seq)),:]
male_plot.iloc[:] = median_values
male_plot['Sex'] = 1
male_plot['Hrate'] = hrate_seq

female_plot = male_plot.copy()
female_plot['Sex'] = 0

predict_male = lr_full.fit(disp=0).get_prediction(male_plot).summary_frame(alpha=0.05)
predict_female = lr_full.fit(disp=0).get_prediction(female_plot).summary_frame(alpha=0.05)


plt.fill_between(male_plot['Hrate'], predict_male['ci_upper'],\
                 predict_male['ci_lower'], alpha=.25,color = 'blue')
plt.plot(male_plot['Hrate'],predict_male['predicted'], label='Male',color = 'blue')
plt.fill_between(female_plot['Hrate'], predict_female['ci_upper'],\
                 predict_female['ci_lower'], alpha=.25,color = 'red')
plt.plot(female_plot['Hrate'],predict_female['predicted'], label='Female',color = 'red')
plt.xlabel('Hrate')
plt.ylabel('Predicted Probability of TCHD')
plt.legend();
```


    
![png](First_circle_prevon_2_files/output_46_0.png)
    



```python
# Gluc & sex
gluc_seq = np.array([*range(40,350,10)])

male_plot = framingham_complete.copy().iloc[range(len(gluc_seq)),:]
male_plot.iloc[:] = median_values
male_plot['Sex'] = 1
male_plot['Gluc'] = gluc_seq

female_plot = male_plot.copy()
female_plot['Sex'] = 0

predict_male = lr_full.fit(disp=0).get_prediction(male_plot).summary_frame(alpha=0.05)
predict_female = lr_full.fit(disp=0).get_prediction(female_plot).summary_frame(alpha=0.05)


plt.fill_between(male_plot['Gluc'], predict_male['ci_upper'],\
                 predict_male['ci_lower'], alpha=.25,color = 'blue')
plt.plot(male_plot['Gluc'],predict_male['predicted'], label='Male',color = 'blue')
plt.fill_between(female_plot['Gluc'], predict_female['ci_upper'],\
                 predict_female['ci_lower'], alpha=.25,color = 'red')
plt.plot(female_plot['Gluc'],predict_female['predicted'], label='Female',color = 'red')
plt.xlabel('Gluc')
plt.ylabel('Predicted Probability of TCHD')
plt.legend();
```


    
![png](First_circle_prevon_2_files/output_47_0.png)
    



```python
# age & sex & meds 
age_seq = np.array([*range(80,145,5)])/2

male_plot = framingham_complete.copy().iloc[range(len(age_seq)),:]
male_plot.iloc[:] = median_values
male_plot['Sex'] = 1
male_plot['Age'] = age_seq

female_plot = male_plot.copy()
female_plot['Sex'] = 0

predict_male = lr_full.fit(disp=0).get_prediction(male_plot).summary_frame(alpha=0.05)
predict_female = lr_full.fit(disp=0).get_prediction(female_plot).summary_frame(alpha=0.05)


fig, axs = plt.subplots(1,2)
axs[0].plot(male_plot['Age'],predict_male['predicted'], label='Male',color = 'blue')
axs[0].plot(female_plot['Age'],predict_female['predicted'], label='Female',color = 'red')
axs[0].fill_between(male_plot['Age'], predict_male['ci_upper'],\
                 predict_male['ci_lower'], alpha=.25,color = 'blue')
axs[0].fill_between(female_plot['Age'], predict_female['ci_upper'],\
                 predict_female['ci_lower'], alpha=.25,color = 'red')
axs[0].set_xlabel('Age')
axs[0].set_ylabel('Predicted Probability of TCHD')
axs[0].set_title('Meds = 0')
axs[0].legend()

male_plot['Meds'] = 1
female_plot['Meds'] = 1
predict_male = lr_full.fit(disp=0).get_prediction(male_plot).summary_frame(alpha=0.05)
predict_female = lr_full.fit(disp=0).get_prediction(female_plot).summary_frame(alpha=0.05)

axs[1].plot(male_plot['Age'],predict_male['predicted'], label='Male',color = 'blue')
axs[1].plot(female_plot['Age'],predict_female['predicted'], label='Female',color = 'red')
axs[1].fill_between(male_plot['Age'], predict_male['ci_upper'],\
                 predict_male['ci_lower'], alpha=.25,color = 'blue')
axs[1].fill_between(female_plot['Age'], predict_female['ci_upper'],\
                 predict_female['ci_lower'], alpha=.25,color = 'red');
axs[1].set_xlabel('Age')
axs[1].set_title('Meds = 1')
axs[1].legend();
```


    
![png](First_circle_prevon_2_files/output_48_0.png)
    



```python
# age & sex & stroke 
age_seq = np.array([*range(80,145,5)])/2

male_plot = framingham_complete.copy().iloc[range(len(age_seq)),:]
male_plot.iloc[:] = median_values
male_plot['Sex'] = 1
male_plot['Age'] = age_seq

female_plot = male_plot.copy()
female_plot['Sex'] = 0

predict_male = lr_full.fit(disp=0).get_prediction(male_plot).summary_frame(alpha=0.05)
predict_female = lr_full.fit(disp=0).get_prediction(female_plot).summary_frame(alpha=0.05)


fig, axs = plt.subplots(1,2)
axs[0].plot(male_plot['Age'],predict_male['predicted'], label='Male',color = 'blue')
axs[0].plot(female_plot['Age'],predict_female['predicted'], label='Female',color = 'red')
axs[0].fill_between(male_plot['Age'], predict_male['ci_upper'],\
                 predict_male['ci_lower'], alpha=.25,color = 'blue')
axs[0].fill_between(female_plot['Age'], predict_female['ci_upper'],\
                 predict_female['ci_lower'], alpha=.25,color = 'red')
axs[0].set_xlabel('Age')
axs[0].set_ylabel('Predicted Probability of TCHD')
axs[0].set_title('Stroke = 0')
axs[0].legend()

male_plot['Stroke'] = 1
female_plot['Stroke'] = 1
predict_male = lr_full.fit(disp=0).get_prediction(male_plot).summary_frame(alpha=0.05)
predict_female = lr_full.fit(disp=0).get_prediction(female_plot).summary_frame(alpha=0.05)

axs[1].plot(male_plot['Age'],predict_male['predicted'], label='Male',color = 'blue')
axs[1].plot(female_plot['Age'],predict_female['predicted'], label='Female',color = 'red')
axs[1].fill_between(male_plot['Age'], predict_male['ci_upper'],\
                 predict_male['ci_lower'], alpha=.25,color = 'blue')
axs[1].fill_between(female_plot['Age'], predict_female['ci_upper'],\
                 predict_female['ci_lower'], alpha=.25,color = 'red');
axs[1].set_xlabel('Age')
axs[1].set_title('Stroke = 1')
axs[1].legend()
```




    <matplotlib.legend.Legend at 0x1ae45528d70>




    
![png](First_circle_prevon_2_files/output_49_1.png)
    



```python
# age & sex & hyp 
age_seq = np.array([*range(80,145,5)])/2

male_plot = framingham_complete.copy().iloc[range(len(age_seq)),:]
male_plot.iloc[:] = median_values
male_plot['Sex'] = 1
male_plot['Age'] = age_seq

female_plot = male_plot.copy()
female_plot['Sex'] = 0

predict_male = lr_full.fit(disp=0).get_prediction(male_plot).summary_frame(alpha=0.05)
predict_female = lr_full.fit(disp=0).get_prediction(female_plot).summary_frame(alpha=0.05)


fig, axs = plt.subplots(1,2)
axs[0].plot(male_plot['Age'],predict_male['predicted'], label='Male',color = 'blue')
axs[0].plot(female_plot['Age'],predict_female['predicted'], label='Female',color = 'red')
axs[0].fill_between(male_plot['Age'], predict_male['ci_upper'],\
                 predict_male['ci_lower'], alpha=.25,color = 'blue')
axs[0].fill_between(female_plot['Age'], predict_female['ci_upper'],\
                 predict_female['ci_lower'], alpha=.25,color = 'red')
axs[0].set_xlabel('Age')
axs[0].set_ylabel('Predicted Probability of TCHD')
axs[0].set_title('Hyp = 0')
axs[0].legend()

male_plot['Hyp'] = 1
female_plot['Hyp'] = 1
predict_male = lr_full.fit(disp=0).get_prediction(male_plot).summary_frame(alpha=0.05)
predict_female = lr_full.fit(disp=0).get_prediction(female_plot).summary_frame(alpha=0.05)

axs[1].plot(male_plot['Age'],predict_male['predicted'], label='Male',color = 'blue')
axs[1].plot(female_plot['Age'],predict_female['predicted'], label='Female',color = 'red')
axs[1].fill_between(male_plot['Age'], predict_male['ci_upper'],\
                 predict_male['ci_lower'], alpha=.25,color = 'blue')
axs[1].fill_between(female_plot['Age'], predict_female['ci_upper'],\
                 predict_female['ci_lower'], alpha=.25,color = 'red');
axs[1].set_xlabel('Age')
axs[1].set_title('Hyp = 1')
axs[1].legend();

```


    
![png](First_circle_prevon_2_files/output_50_0.png)
    



```python
# age & sex & diab 
age_seq = np.array([*range(80,145,5)])/2

male_plot = framingham_complete.copy().iloc[range(len(age_seq)),:]
male_plot.iloc[:] = median_values
male_plot['Sex'] = 1
male_plot['Age'] = age_seq

female_plot = male_plot.copy()
female_plot['Sex'] = 0

predict_male = lr_full.fit(disp=0).get_prediction(male_plot).summary_frame(alpha=0.05)
predict_female = lr_full.fit(disp=0).get_prediction(female_plot).summary_frame(alpha=0.05)


fig, axs = plt.subplots(1,2)
axs[0].plot(male_plot['Age'],predict_male['predicted'], label='Male',color = 'blue')
axs[0].plot(female_plot['Age'],predict_female['predicted'], label='Female',color = 'red')
axs[0].fill_between(male_plot['Age'], predict_male['ci_upper'],\
                 predict_male['ci_lower'], alpha=.25,color = 'blue')
axs[0].fill_between(female_plot['Age'], predict_female['ci_upper'],\
                 predict_female['ci_lower'], alpha=.25,color = 'red')
axs[0].set_xlabel('Age')
axs[0].set_ylabel('Predicted Probability of TCHD')
axs[0].set_title('Diab = 0')
axs[0].legend()

male_plot['Diab'] = 1
female_plot['Diab'] = 1
predict_male = lr_full.fit(disp=0).get_prediction(male_plot).summary_frame(alpha=0.05)
predict_female = lr_full.fit(disp=0).get_prediction(female_plot).summary_frame(alpha=0.05)

axs[1].plot(male_plot['Age'],predict_male['predicted'], label='Male',color = 'blue')
axs[1].plot(female_plot['Age'],predict_female['predicted'], label='Female',color = 'red')
axs[1].fill_between(male_plot['Age'], predict_male['ci_upper'],\
                 predict_male['ci_lower'], alpha=.25,color = 'blue')
axs[1].fill_between(female_plot['Age'], predict_female['ci_upper'],\
                 predict_female['ci_lower'], alpha=.25,color = 'red');
axs[1].set_xlabel('Age')
axs[1].set_title('Diab = 1')
axs[1].legend();
```


    
![png](First_circle_prevon_2_files/output_51_0.png)
    


Looking at the plots, we can notice that **Sex**, **Age**, **Cig**, **Chol**,  **SysP**,  **DiaP**, and **Gluc**  (predictors that were 'significant') seem to have a noticeable effect on the probability of TCHD. Interestingly enough, **DiaP** is the only numerical predictor that appears to have a strong nonlinear effect (nonlinearity of the effect of **Cig** seems to be caused by the lack of observations for **Cig** > 50; in our R implementation, where we used *restricted* cubic splines, the effect of **Cig** is estimated as monotonically increasing ).

Factors **Stroke** and **Diab** appear to have an effect, but the prediction uncertainty is too high (probably due to the low number of cases, as we discussed earlier). Variables **Edu**, **Meds**, **Hyp**, **BMI**, and **Hrate** seem to have very little effect.

Next, let us check the significance of the interactions.


```python
# test interactions
lr_no_inter = smf.logit(formula='TCHD ~ Sex + bs(Age, df=3) + C(Edu, Poly) + bs(Cig, df=3) + Meds + Stroke + Hyp + Diab + bs(Chol, df=3) + \
                     bs(SysP, df=3) + bs(DiaP, df=3) + bs(BMI, df=3) + bs(Hrate, df=3) + bs(Gluc, df=3)', \
                     data=framingham_complete)

1 - chi2.cdf(2*(lr_full_fit.llf-lr_no_inter.fit(disp=0).llf),(lr_full.df_model - lr_no_inter.df_model))
```




    0.527900655708825



Interactions do not seem significant in the model. Let us have a look at the nonlinear terms. 


```python
# test nonlinearities
lr_lin = smf.logit(formula='TCHD ~ Sex + Age + C(Edu, Poly) + Cig + Meds + Stroke + Hyp + Diab + Chol + \
                     SysP + DiaP + BMI + Hrate + Gluc + \
                     Age:(Cig + Stroke + Hyp + Diab + Chol + SysP + DiaP + BMI + Hrate + Gluc) + \
                     Sex:(Cig + Stroke + Hyp + Diab + Chol + SysP + DiaP + BMI + Hrate + Gluc)', data=framingham_complete)

1 - chi2.cdf(2*(lr_full_fit.llf-lr_lin.fit(disp=0).llf),(lr_full.df_model - lr_lin.df_model))
```




    0.004021774649243026



 The model's nonlinear termsl are significant. Let us determine for which variable it is.


```python
LR_test2 = pd.DataFrame(index=['Age','Cig','Chol','SysP','DiabP','BMI','Hrate','Gluc'], \
                       columns = ['Deviance','DoF','P-value'])


lr_lin_age = smf.logit(formula='TCHD ~ Sex + Age + C(Edu, Poly) + bs(Cig, df=3) + Meds + Stroke + Hyp + Diab + bs(Chol, df=3) + \
                     bs(SysP, df=3) + bs(DiaP, df=3) + bs(BMI, df=3) + bs(Hrate, df=3) + bs(Gluc, df=3) + \
                     Age:(Cig + Stroke + Hyp + Diab + Chol + SysP + DiaP + BMI + Hrate + Gluc) + \
                     Sex:(Cig + Stroke + Hyp + Diab + Chol + SysP + DiaP + BMI + Hrate + Gluc)', data=framingham_complete)
                     
LR_test2.iloc[0,0] = 2*(lr_full_fit.llf-lr_lin_age.fit(disp=0).llf)
LR_test2.iloc[0,1] = (lr_full.df_model - lr_lin_age.df_model)
LR_test2.iloc[0,2] = 1 - chi2.cdf(LR_test2.iloc[0,0],LR_test2.iloc[0,1])


lr_lin_cig = smf.logit(formula='TCHD ~ Sex + bs(Age, df=3) + C(Edu, Poly) + Cig + Meds + Stroke + Hyp + Diab + bs(Chol, df=3) + \
                     bs(SysP, df=3) + bs(DiaP, df=3) + bs(BMI, df=3) + bs(Hrate, df=3) + bs(Gluc, df=3) + \
                     Age:(Cig + Stroke + Hyp + Diab + Chol + SysP + DiaP + BMI + Hrate + Gluc) + \
                     Sex:(Cig + Stroke + Hyp + Diab + Chol + SysP + DiaP + BMI + Hrate + Gluc)', data=framingham_complete)
                     
LR_test2.iloc[1,0] = 2*(lr_full_fit.llf-lr_lin_cig.fit(disp=0).llf)
LR_test2.iloc[1,1] = (lr_full.df_model - lr_lin_cig.df_model)
LR_test2.iloc[1,2] = 1 - chi2.cdf(LR_test2.iloc[1,0],LR_test2.iloc[1,1])



lr_lin_chol = smf.logit(formula='TCHD ~ Sex + bs(Age, df=3) + C(Edu, Poly) + bs(Cig, df=3) + Meds + Stroke + Hyp + Diab + Chol + \
                     bs(SysP, df=3) + bs(DiaP, df=3) + bs(BMI, df=3) + bs(Hrate, df=3) + bs(Gluc, df=3) + \
                     Age:(Cig + Stroke + Hyp + Diab + Chol + SysP + DiaP + BMI + Hrate + Gluc) + \
                     Sex:(Cig + Stroke + Hyp + Diab + Chol + SysP + DiaP + BMI + Hrate + Gluc)', data=framingham_complete)
                     
LR_test2.iloc[2,0] = 2*(lr_full_fit.llf-lr_lin_chol.fit(disp=0).llf)
LR_test2.iloc[2,1] = (lr_full.df_model - lr_lin_chol.df_model)
LR_test2.iloc[2,2] = 1 - chi2.cdf(LR_test2.iloc[2,0],LR_test2.iloc[2,1])


lr_lin_sysp = smf.logit(formula='TCHD ~ Sex + bs(Age, df=3) + C(Edu, Poly) + bs(Cig, df=3) + Meds + Stroke + Hyp + Diab + bs(Chol, df=3) + \
                     SysP + bs(DiaP, df=3) + bs(BMI, df=3) + bs(Hrate, df=3) + bs(Gluc, df=3) + \
                     Age:(Cig + Stroke + Hyp + Diab + Chol + SysP + DiaP + BMI + Hrate + Gluc) + \
                     Sex:(Cig + Stroke + Hyp + Diab + Chol + SysP + DiaP + BMI + Hrate + Gluc)', data=framingham_complete)
                     
LR_test2.iloc[3,0] = 2*(lr_full_fit.llf-lr_lin_sysp.fit(disp=0).llf)
LR_test2.iloc[3,1] = (lr_full.df_model - lr_lin_sysp.df_model)
LR_test2.iloc[3,2] = 1 - chi2.cdf(LR_test2.iloc[3,0],LR_test2.iloc[3,1])


lr_lin_diap = smf.logit(formula='TCHD ~ Sex + bs(Age, df=3) + C(Edu, Poly) + bs(Cig, df=3) + Meds + Stroke + Hyp + Diab + bs(Chol, df=3) + \
                     bs(SysP, df=3) + DiaP + bs(BMI, df=3) + bs(Hrate, df=3) + bs(Gluc, df=3) + \
                     Age:(Cig + Stroke + Hyp + Diab + Chol + SysP + DiaP + BMI + Hrate + Gluc) + \
                     Sex:(Cig + Stroke + Hyp + Diab + Chol + SysP + DiaP + BMI + Hrate + Gluc)', data=framingham_complete)
                     
LR_test2.iloc[4,0] = 2*(lr_full_fit.llf-lr_lin_diap.fit(disp=0).llf)
LR_test2.iloc[4,1] = (lr_full.df_model - lr_lin_diap.df_model)
LR_test2.iloc[4,2] = 1 - chi2.cdf(LR_test2.iloc[4,0],LR_test2.iloc[4,1])



lr_lin_bmi = smf.logit(formula='TCHD ~ Sex + bs(Age, df=3) + C(Edu, Poly) + bs(Cig, df=3) + Meds + Stroke + Hyp + Diab + bs(Chol, df=3) + \
                     bs(SysP, df=3) + bs(DiaP, df=3) + BMI + bs(Hrate, df=3) + bs(Gluc, df=3) + \
                     Age:(Cig + Stroke + Hyp + Diab + Chol + SysP + DiaP + BMI + Hrate + Gluc) + \
                     Sex:(Cig + Stroke + Hyp + Diab + Chol + SysP + DiaP + BMI + Hrate + Gluc)', data=framingham_complete)
                     
LR_test2.iloc[5,0] = 2*(lr_full_fit.llf-lr_lin_bmi.fit(disp=0).llf)
LR_test2.iloc[5,1] = (lr_full.df_model - lr_lin_bmi.df_model)
LR_test2.iloc[5,2] = 1 - chi2.cdf(LR_test2.iloc[5,0],LR_test2.iloc[5,1])


lr_lin_hrate = smf.logit(formula='TCHD ~ Sex + bs(Age, df=3) + C(Edu, Poly) + bs(Cig, df=3) + Meds + Stroke + Hyp + Diab + bs(Chol, df=3) + \
                     bs(SysP, df=3) + bs(DiaP, df=3) + bs(BMI, df=3) + Hrate + bs(Gluc, df=3) + \
                     Age:(Cig + Stroke + Hyp + Diab + Chol + SysP + DiaP + BMI + Hrate + Gluc) + \
                     Sex:(Cig + Stroke + Hyp + Diab + Chol + SysP + DiaP + BMI + Hrate + Gluc)', data=framingham_complete)
                     
LR_test2.iloc[6,0] = 2*(lr_full_fit.llf-lr_lin_hrate.fit(disp=0).llf)
LR_test2.iloc[6,1] = (lr_full.df_model - lr_lin_hrate.df_model)
LR_test2.iloc[6,2] = 1 - chi2.cdf(LR_test2.iloc[6,0],LR_test2.iloc[6,1])


lr_lin_gluc = smf.logit(formula='TCHD ~ Sex + bs(Age, df=3) + C(Edu, Poly) + bs(Cig, df=3) + Meds + Stroke + Hyp + Diab + bs(Chol, df=3) + \
                     bs(SysP, df=3) + bs(DiaP, df=3) + bs(BMI, df=3) + bs(Hrate, df=3) + Gluc + \
                     Age:(Cig + Stroke + Hyp + Diab + Chol + SysP + DiaP + BMI + Hrate + Gluc) + \
                     Sex:(Cig + Stroke + Hyp + Diab + Chol + SysP + DiaP + BMI + Hrate + Gluc)', data=framingham_complete)
                     
LR_test2.iloc[7,0] = 2*(lr_full_fit.llf-lr_lin_gluc.fit(disp=0).llf)
LR_test2.iloc[7,1] = (lr_full.df_model - lr_lin_gluc.df_model)
LR_test2.iloc[7,2] = 1 - chi2.cdf(LR_test2.iloc[7,0],LR_test2.iloc[7,1])


LR_test2
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Deviance</th>
      <th>DoF</th>
      <th>P-value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Age</th>
      <td>2.447749</td>
      <td>2.0</td>
      <td>0.294088</td>
    </tr>
    <tr>
      <th>Cig</th>
      <td>4.367284</td>
      <td>2.0</td>
      <td>0.112631</td>
    </tr>
    <tr>
      <th>Chol</th>
      <td>6.652814</td>
      <td>2.0</td>
      <td>0.035922</td>
    </tr>
    <tr>
      <th>SysP</th>
      <td>1.783007</td>
      <td>2.0</td>
      <td>0.410039</td>
    </tr>
    <tr>
      <th>DiabP</th>
      <td>11.663551</td>
      <td>2.0</td>
      <td>0.002933</td>
    </tr>
    <tr>
      <th>BMI</th>
      <td>0.929204</td>
      <td>2.0</td>
      <td>0.628385</td>
    </tr>
    <tr>
      <th>Hrate</th>
      <td>1.120257</td>
      <td>2.0</td>
      <td>0.571136</td>
    </tr>
    <tr>
      <th>Gluc</th>
      <td>4.585019</td>
      <td>2.0</td>
      <td>0.101013</td>
    </tr>
  </tbody>
</table>
</div>



Notably, it is the effect of **DiaP** which has a highly nonlinear dependence that seems to be significant. The 'J-shaped' dependence of the probability of developing heart disease (probability seems to increase at both low and high levels of diastolic pressure) is a known pattern observed in medical data. However, nowadays, the consensus is that the shape is a result of confounding and reverse causation (i.e., low blood pressure can indicate a high-risk state caused by other comorbidities), and it is highly unlikely to reflect a causal process [[3](#3)].

### Model Diagnostics <a class="anchor" id="diagnostics"></a>

Logistic regression of a binary response does not have distributional assumptions (we directly model the probability of an event $\mathrm{ln} \frac{p}{1-p} = X\beta$); hence, bias in our estimates will be connected to model misspecifications, such as omitted variable bias or the choice of the link function (which is in our case *logit*: $\mathrm{logit} (p) = \mathrm{ln} \frac{p}{1-p}$). 

An interesting fact about logistic regression is that omitted variable bias is caused by both missing predictors correlated with $X$ (as in linear regression) but also by uncorrelated omitted variables (unlike linear regression) [[4](#4)]. However, this second source of bias is always downwards (i.e., other effects will tend to look smaller than they actually are).

A standard method for assessing model misspecification in linear regression is to analyze residuals. However, plain residual plots are much less helpful in binary regression. 


```python
# raw residuals
plt.scatter(lr_full_fit.predict(), framingham_complete['TCHD'] - lr_full_fit.predict())
plt.title('Raw Residuals vs Predicted Probabilites')
plt.xlabel('Predicted Probabilites')
plt.ylabel('Raw Residuals');
```


    
![png](First_circle_prevon_2_files/output_60_0.png)
    



```python
sm.qqplot( framingham_complete['TCHD'] - lr_full_fit.predict(), line='s');
```


    
![png](First_circle_prevon_2_files/output_61_0.png)
    


Here, we plotted so-called *raw residuals* (observed outcomes minus predicted probabilities of outcomes) vs. predicted probabilities. These residuals have values in the interval [-1,1] that are quite apparently not normally distributed (and they cannot be, since apart from being bounded to [-1,1], they are inherently heteroskedastic, since the variance of binary outcome is $p(1-p)$). 

The *Pearson residuals* (raw residuals divided by their expected deviance) are as follows.


```python
# Pearson residuals
pearson = pd.DataFrame(index = [*range(len(lr_full_fit.predict() ))], columns= ['Pearson'])
pearson['Pearson'] = ((framingham_complete['TCHD'] - \
                      lr_full_fit.predict())/np.sqrt(lr_full_fit.predict() * (1-lr_full_fit.predict())))
    
plt.scatter(lr_full_fit.predict(), pearson['Pearson'])
plt.title('Pearson Residuals vs Predicted Probabilites')
plt.xlabel('Pearson Probabilites')
plt.ylabel('Deviance Residuals');
```


    
![png](First_circle_prevon_2_files/output_63_0.png)
    



```python
sm.qqplot(pearson['Pearson'], line='s');
```


    
![png](First_circle_prevon_2_files/output_64_0.png)
    


The last residual we will plot here are the *deviance residuals* which are based on the individual contributions to the log-likelihood of the model. 


```python
# Deviance residuals
plt.scatter(lr_full_fit.predict(), lr_full_fit.resid_dev)
plt.title('Deviance Residuals vs Predicted Probabilites')
plt.xlabel('Predicted Probabilites')
plt.ylabel('Deviance Residuals')
```




    Text(0, 0.5, 'Deviance Residuals')




    
![png](First_circle_prevon_2_files/output_66_1.png)
    



```python
sm.qqplot(lr_full_fit.resid_dev, line='s');
```


    
![png](First_circle_prevon_2_files/output_67_0.png)
    


None of these popular residuals are close to being normal. The problem is the fact that the binary data does not meet the so-called *small dispersion asymptotics* which is required for deviance/Pearson residuals to be approximately normal (if we interpret binary data as counts, we would essentially need that every observation has at least observed 3 'successes' for small dispersion asymptotics to hold) [[5](#5)].

This observation, however, provides us with a method to get residuals that are approximately normal. If we aggregate the data by the values of their linear predictor $X\beta$, we obtain groups with a sufficient number of successes in each of them for the asymptotics to hold. 


```python
pearson['Lin_pred'] = lr_full_fit.predict(which = 'linear')
# group Pearson residuals by quantiles of linear predictor and calculate mean Pearson residual for each group
sm.qqplot(pearson.groupby(pd.qcut(pearson['Lin_pred'], 50, labels=False)).mean()['Pearson'], line='s'); 
```


    
![png](First_circle_prevon_2_files/output_69_0.png)
    


 An alternative approach that we explored in R is to use so-called *quantile residuals* (https://cran.r-project.org/web/packages/DHARMa/vignettes/DHARMa.html#lm-and-glm). These are based on a simulation approach similar to a parametric bootstrap; see The Second Circle: Logistic Regression, Part One for more details.

### Measures of Predictive Performance <a class="anchor" id="performance"></a>

Let us evaluate our model's performance on new data. It is expected that this model will perform worse at predicting the probability of **TCHD** for new data. The following heuristic van Houwelingen and Le Cessie *shrinkage* estimate [[6](#6)] can often estimate the decrease in performance quite well.


```python
# trivial model
lr_triv = smf.logit(formula='TCHD ~ 1',data=framingham_complete)
lr_triv_fit = lr_triv.fit(disp=0) 
# deviance
dev = 2*(lr_full_fit.llf - lr_triv_fit.llf)
# shrinkage
(dev-len(lr_full_fit.params))/dev
```




    0.8734446169776853



The shrinkage factor is 0.87. Thus, we expect the model to perform approximately 13% worse on new data. The model overfits a bit too much (the rule of thumb is 10% shrinkage [[1](#1)]). To reduce overfitting somewhat, we can use the observation that interactions in our model appeared largely insignificant.


```python
lr_final = smf.logit(formula='TCHD ~ Sex + bs(Age, df=3) + C(Edu, Poly) + bs(Cig, df=3)  + Meds + Stroke + Hyp + Diab + bs(Chol, df=3) + \
                     bs(SysP, df=3) + bs(DiaP, df=3) + bs(BMI, df=3) + bs(Hrate, df=3) + bs(Gluc, df=3)',\
                     data=framingham_complete)
lr_final_fit = lr_final.fit(disp=0)  

dev = 2*(lr_final_fit.llf - lr_triv_fit.llf)
(dev-len(lr_final_fit.params))/dev
```




    0.9174757222967561



This value of estimated shrinkage is much more acceptable. Consequently, we will consider the model without interactions as our final model for predicting the probabilities of **TCHD**.

#### Prediction Accuracy (percentage of correct predictions) <a class="anchor" id="accuracy"></a>

Let us start the evaluation of the predictive performance of the model. The most natural performance metric is overall prediction accuracy. To do so, we first select a probability threshold and then predict that a particular subject will develop **CHD** in ten years if the estimated probability exceeds the threshold.  These predictions are often reported in a so-called *confusion table*, which splits the results into four categories: true positive, false positive, true negative, and false negative. The overall prediction accuracy is then the percentage of true positives and true negatives.



```python
from sklearn.metrics import (confusion_matrix, accuracy_score,\
                             roc_curve, roc_auc_score, log_loss, brier_score_loss)

# predictions
y_pred_prob = lr_final_fit.predict()
y_obs = framingham_complete['TCHD'].to_numpy()

# confusion table for 0.5 threshold
y_pred = (y_pred_prob > 0.5).astype(int)
confusion_matrix(y_obs,y_pred)
```




    array([[3074,   25],
           [ 503,   54]])




```python
# prediction accuracy
accuracy_score(y_obs,y_pred)
```




    0.8555798687089715



The overall accuracy is about 85%; however, this value depends on the threshold. Let's compare the accuracy of predictions for various thresholds.


```python
# accuracy
accuracy_score(y_obs,y_pred)

accuracy_scores = np.zeros(21)
accuracy_thresholds = np.zeros(21)

for k in range(21):
    accuracy_thresholds[k] = k/20
    accuracy_scores[k] = accuracy_score(y_obs,(y_pred_prob > accuracy_thresholds[k]).astype(int))
plt.scatter(accuracy_thresholds,accuracy_scores);
plt.xlabel('Probability Threshold')
plt.ylabel('Accuracy')
```




    Text(0, 0.5, 'Accuracy')




    
![png](First_circle_prevon_2_files/output_80_1.png)
    


We see that the accuracy peaks at about the aforementioned 85%. 

There are some serious issues with using the overall accuracy as the performance index. Accuracy is not a proper so-called *scoring rule* (its value is not optimal for the true probability distribution of the outcome). Thus, the model that maximizes accuracy might not correctly model the underlying probability distribution of the outcomes. Accuracy ignores the uncertainty of predictions (it does not care whether the predicted probability of the outcome was 1%, 51%, or 99%; it only cares whether the predicted outcome at a given threshold is correct). 

It is also misleading for an imbalanced prevalence. Think of the dataset with 99% negative results and 1% positive results. In such a dataset, predicting only negative values reaches 99% accuracy. However, the overall accuracy can yield weird recommendations, even for balanced data (see https://www.fharrell.com/post/class-damage/, where a model that ignored an important predictor had better accuracy). 

The last issue we mention here is its dependence on the threshold. As we will see later, selecting a threshold is tied to decision-making (and to considering the costs of those decisions).
<br/>

#### Specificity & Sensitivity, ROC curve <a class="anchor" id="roc"></a>

Since the prevalence heavily influences the accuracy, there are two other popular metrics tied to the outcomes: specificity (true negative rate) and sensitivity (true positive rate). 


```python
specificity = np.zeros(21)
sensitivity = np.zeros(21)

for k in range(21):
    conf_table = confusion_matrix(y_obs,(y_pred_prob > k/20).astype(int))
    specificity[k] = conf_table[0,0]/(conf_table[0,0] + conf_table[0,1])
    sensitivity[k] = conf_table[1,1]/(conf_table[1,0] + conf_table[1,1])
    
plt.scatter(accuracy_thresholds,specificity)
plt.xlabel('Probability Threshold')
plt.ylabel('Specificity');
```


    
![png](First_circle_prevon_2_files/output_83_0.png)
    



```python
plt.scatter(accuracy_thresholds,sensitivity)
plt.xlabel('Probability Threshold')
plt.ylabel('Sensitivity');
```


    
![png](First_circle_prevon_2_files/output_84_0.png)
    


 We see that our model is not very sensitive; we need to choose a relatively low threshold to detect the majority of positive cases (those that actually developed **CHD**), which results in an overall high false positive rate (low specificity). 


```python
# confusion table for 0.1 threshold
y_pred = (y_pred_prob > 0.1).astype(int)
confusion_matrix(y_obs,y_pred)
```




    array([[1495, 1604],
           [  90,  467]])



<br/> Still, we predict about half of the negative cases (those that did not develop *CHD* in ten years) correctly, which could make the model still valuable enough for making decisions (see decision curve analysis later in the text). This analysis provides an example of a typical trade-off between sensitivity and specificity, often depicted for a particular model of binary response using the ROC (Receiver Operating Characteristic) curve.


```python
plt.scatter(1-specificity,sensitivity);
plt.xlabel('1 - Specificity (false positive rate)')
plt.ylabel('Sensitivity (true positive rate)');
```


    
![png](First_circle_prevon_2_files/output_88_0.png)
    



```python
# or  using roc_curve function
fpr, tpr, thresholds = roc_curve(y_obs,y_pred_prob)
plt.scatter(fpr,tpr)
plt.xlabel('1 - Specificity (false positive rate)')
plt.ylabel('Sensitivity (true positive rate)');
```


    
![png](First_circle_prevon_2_files/output_89_0.png)
    


The area under the ROC curve () is another popular performance metric for binary classifiers.


```python
roc_auc_score(y_obs,y_pred_prob)
```




    0.747502379582688



A ROC curve area of 0.5 corresponds to random predictions, and a value of 1 indicates perfect predictions (i.e., the model perfectly separates positive and negative results). Generally, a value greater than 0.7 is considered an acceptable predictive power. 

The area under ROC for binary classification equals the *concordance index*. The concordance index is equal to the concordance probability: the proportion of *comparable pairs* in the data (pairs that have different outcomes) that are concordant (individuals with the higher predicted probability experienced the event). 

*c*-index is another improper scoring rule, and thus, *c* can give some weird results [[7](#7)]. It is also a somewhat insensitive metric (adding or removing significant predictors may not change the value of *c*), and thus, it is overall not sufficient for comparing models.

#### Strictly Proper Scoring Rules (logarithmic score and Brier score) <a class="anchor" id="scoring"></a>

The first strictly proper scoring rule we were actually using throughout this project is the logarithimic scoring rule $\sum y_i\mathrm{log} p_i + (1-y_i)\mathrm{log} (1-p_i)$.


```python
(y_obs*np.log(y_pred_prob) + (1-y_obs)*np.log(1-y_pred_prob)).sum()
```




    -1360.3218381906488



We observe that this is nothing but the log-likelihood $\mathrm{log}L$ of the logistic model (i.e., this is the scoring rule that the logistic regression coefficients maximize).


```python
lr_final_fit.llf
```




    -1360.3218381906488



We can also compute logarithimic scoring rule (multiplied by $-\frac{1}{n}$) using the function *log_loss*.


```python
log_loss(y_obs, y_pred_prob)
```




    0.3720792774044444




```python
-(y_obs*np.log(y_pred_prob) + (1-y_obs)*np.log(1-y_pred_prob)).mean()
```




    0.3720792774044444



We used the logarithmic scoring rule to compare nested models throughout this presentation via a likelihood ratio test. Alternatively, we could use Akaike information criterion, which is $\mathrm{AIC} = 2k - 2 \mathrm{log} L$. For example, using AIC to compare our full model with the model without interactions, we would get the difference


```python
lr_full_fit.aic  - lr_final_fit.aic # AIC full model - AIC model without interactions
```




    21.093342730696804



which is significantly greater than 10, indicating that our full model has essentially no support according to the rules-of-thumbs from [[8](#8)].* The logarithmic scoring rule and AIC can be used only to compare models on a particular dataset. Their total value has little meaning. 

Another strictly proper scoring rule commonly used is the Brier score. The Brier score is a mean square error between the observed outcomes (0 and 1) and the estimated probabilities.


```python
((y_obs-y_pred_prob)**2).mean()
```




    0.11335099671704732




```python
brier_score_loss(y_obs,y_pred_prob)
```




    0.11335099671704732



There are two interesting values of estimated probabilities for which we can compute the Brier score: 0.5 and the mean of the observed outcome.


```python
((y_obs-0.5)**2).mean()
```




    0.25




```python
((y_obs-y_obs.mean())**2).mean()
```




    0.12914107501113242



The Brier score for "coin flip" predictions is always 0.25. The second Brier score corresponds to predicting the outcome based on overall prevalence. We see that our model is better than that

The optimal value of the Brier score depends on the actual distribution. Consequently, the absolute value of the Brier score may be a bit misleading, depending on the distribution of the outcome [[9](#9)]. However, the Brier score is easier to interpret than the logarithmic scoring rule overall, and its values are somewhat comparable across models and data sets when the underlying distribution is the same. 

#### Calibration <a class="anchor" id="calibration"></a>

The model's predicted probabilities should correspond to the actual probabilities. Otherwise, the model might be too confident in its predictions. Logistic regression models are typically well-calibrated because they optimize the logarithmic scoring rule.

The simplest evaluation of calibration is to split the data by predicted probability (e.g., deciles) and compare the estimated mean probabilities with the observed proportions. 


```python
from sklearn.calibration import calibration_curve  
prob_true, prob_pred = calibration_curve(y_obs,y_pred_prob, n_bins=10, strategy = 'quantile')
plt.scatter(prob_true,prob_pred);
plt.xlabel('Observed probability')
plt.ylabel('Predicted Probability');
```


    
![png](First_circle_prevon_2_files/output_110_0.png)
    



```python
sm.OLS(endog = prob_pred, exog = pd.DataFrame(prob_true).assign(const=1)).fit(disp=0).params.iloc[0] # OLS fit using the calibration data
```




    1.012702800020864



We see that the points are near the diagonal. Since logistic regression is usually well-calibrated on the training data, we can evaluate calibration by fitting a logistic regression model to the predicted probabilities.


```python
logitp = np.log(y_pred_prob/(1-y_pred_prob)) # logit p = log p/(1-p)
calib_model_fit = sm.Logit(endog= framingham_complete['TCHD'],exog = pd.DataFrame(logitp).assign(const=1)).fit(disp=0) # fit logit
print(calib_model_fit.summary())
```

                               Logit Regression Results                           
    ==============================================================================
    Dep. Variable:                   TCHD   No. Observations:                 3656
    Model:                          Logit   Df Residuals:                     3654
    Method:                           MLE   Df Model:                            1
    Date:                Thu, 04 Dec 2025   Pseudo R-squ.:                  0.1281
    Time:                        12:25:35   Log-Likelihood:                -1360.3
    converged:                       True   LL-Null:                       -1560.3
    Covariance Type:            nonrobust   LLR p-value:                 5.842e-89
    ==============================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    0              1.0000      0.056     17.962      0.000       0.891       1.109
    const      -5.787e-16      0.095  -6.06e-15      1.000      -0.187       0.187
    ==============================================================================
    

Since we used a logistic model in the first place, this fit naturally has a slope of one. On the new data, this slope would, however, be different from one in general, indicating that the model may no longer be as well calibrated on the new data. 

#### Decision Curve Analysis <a class="anchor" id="dca"></a>

The previously mentioned performance and calibration indices did not assess whether the model is actually good at making decisions. Decision curve analysis (DCA) attempts to remedy that [[10](#10)].

We consider a simple decision problem. Let the probability of the condition (in our case **TCHD**) be $p$. We have four outcomes: true positive, false positive (followed by treatment), false negative, and true negative (followed by no treatment). Let the value of the outcomes be $a, b, c$, and $d$, respectively. Threshold probability $p_t$, when the expected benefit of the treatment is equal to the expected benefit of avoiding the treatment, meets
$p_ta + (1-p_t)b = p_tc + (1-p_t)d$, i.e., $\frac{1-p_t}{p_t} = \frac{a-c}{d-b}$: $d-b$ is a consequence of being treated unnecessarily (false positive result),  $a − c$ is the consequence of avoiding treatment (false negative result). Thus, we should select the threshold probability $p_t$ based on our choice of the ratio $\frac{a-c}{d-b}$ (the ratio of the consequences of a false positive outcome and a false negative outcome).

Let $\hat{p}$ be the estimated probability of the condition using our model. Let's fix the consequence of a false negative result as $a-c = 1$, then the consequence of a false positive result is $d-b = -\frac{\hat{p}_t}{1-\hat{p}_t}$. The net benefit for the threshold probability $\hat{p}_t$ is given as  $$\mathrm{net\;benefit} = \frac{\mathrm{true\; positive\; count} (\hat{p}_t)}{n}  - \frac{\mathrm{false\; positive\; count }(\hat{p}_t)}{n}\frac{\hat{p}_t}{1-\hat{p}_t}$$ where true positive and false positive counts are evaluated based on the threshold $\hat{p}_t$ [[10](#10)].

The DCA is visualized by the following plot comparing the treat-all and treat-none policies with the policy based on predicted probabilities obtained from our model.


```python
dca_curve  = np.zeros(100)
tall_curve =  np.zeros(100)
tnone_curve = np.zeros(100)

negative_c = (framingham_complete['TCHD'] == 0).sum()
positive_c = (framingham_complete['TCHD'] == 1).sum()
dca_thresholds = np.zeros(100)

for k in range(100):
    dca_thresholds[k] = k/100
    conf_table = confusion_matrix(y_obs,(y_pred_prob > dca_thresholds[k]).astype(int))
    dca_curve[k] = 1/len(framingham_complete)*\
        (conf_table[1,1] - conf_table[0,1]*dca_thresholds[k]/(1-dca_thresholds[k]))
    tall_curve[k] = 1/len(framingham_complete)*\
        (positive_c - negative_c*dca_thresholds[k]/(1-dca_thresholds[k]))
    tnone_curve[k] = 0
    
plt.plot(dca_thresholds,dca_curve,color = 'blue')
plt.plot(dca_thresholds,tnone_curve, color = 'red')
plt.plot(dca_thresholds,tall_curve, color = 'green')
plt.ylim(dca_curve[99]-0.01, dca_curve[0]+0.01);
plt.xlabel('Threshold Probability')
plt.ylabel('Net Benefit');
```


    
![png](First_circle_prevon_2_files/output_116_0.png)
    


We see that our model is uniformly better at making decisions in terms of net benefit than the treat-all and treat-none policies. Thus, our model could be helpful in decision-making in practice.

## Model Validation <a class="anchor" id="validation"></a>

We estimated the performance metrics on the complete data. Next, we will estimate their generalizability to new data using cross-validation.


```python
np.random.seed(123)

rep = 100
folds = 10
kf = KFold(n_splits=10) # create folds

metrics_cv =  pd.DataFrame(index=range(rep*folds),columns = ['c-index (AUC)','Brier score','log score','calibration'])
dca_curve_cv  = pd.DataFrame(index=range(rep*folds),columns = range(100))

k = 0
for i in range(rep):
    idx_cv = np.random.choice([*range(len(framingham_complete))],len(framingham_complete), replace=False)
    
    for j, (train_index, test_index) in enumerate(kf.split(idx_cv)):
        
        train_set = idx_cv[train_index]
        test_set = idx_cv[test_index]
        lr_cv_new = smf.logit(formula='TCHD ~ Sex + bs(Age, lower_bound=30, upper_bound = 75, df=3) + C(Edu, Poly) + \
                             bs(Cig, lower_bound= 0, upper_bound = 80, df=3) + Meds + Stroke + Hyp + Diab + bs(Chol, lower_bound= 100, upper_bound = 600, df=3) + \
                             bs(SysP, lower_bound= 80, upper_bound = 300, df=3) + bs(DiaP, lower_bound= 45, upper_bound = 150, df=3) + \
                             bs(BMI, lower_bound= 15, upper_bound = 60, df=3) + bs(Hrate, lower_bound= 40, upper_bound = 150, df=3) + \
                             bs(Gluc, df=3, lower_bound= 40, upper_bound = 400)',
                             data=framingham_complete.iloc[train_set])
        
        y_pred_prob_new = lr_cv_new.fit(disp=0).predict(framingham_complete.iloc[test_set]) 
        y_obs_new = framingham_complete.iloc[test_set]['TCHD']
        logitp_new = np.log(y_pred_prob_new/(1-y_pred_prob_new))
        
        metrics_cv.iloc[k,0] = roc_auc_score(y_obs_new,y_pred_prob_new)
        metrics_cv.iloc[k,1] = brier_score_loss(y_obs_new,y_pred_prob_new)
        metrics_cv.iloc[k,2] = log_loss(y_obs_new,y_pred_prob_new)
        metrics_cv.iloc[k,3] = sm.Logit(endog= y_obs_new,exog = pd.DataFrame(logitp_new).assign(const=1)).fit(disp=0).params.iloc[0]
        
        for n in range(100):
            conf_table_new = confusion_matrix(y_obs_new,(y_pred_prob_new > dca_thresholds[n]).astype(int))
            dca_curve_cv.iloc[k,n] = 1/len(test_set)*\
                (conf_table_new[1,1] - conf_table_new[0,1]*dca_thresholds[n]/(1-dca_thresholds[n]))
            
        k = k + 1
```


```python
metrics_cv.mean()
```




    c-index (AUC)    0.730285
    Brier score      0.116806
    log score        0.383871
    calibration      0.889377
    dtype: object



We observe that the predicted c-index on new data is slightly lower. Analogously, the Brier score and the logarithmic score estimate are a bit higher. The model is also no longer perfectly calibrated. Nevertheless, overall, the performance did not degrade much. 


```python
plt.plot(dca_thresholds,dca_curve_cv.mean(),color = 'blue')
plt.plot(dca_thresholds,tnone_curve, color = 'red')
plt.plot(dca_thresholds,tall_curve, color = 'green')
plt.ylim(dca_curve[99]-0.01, dca_curve[0]+0.01);
```


    
![png](First_circle_prevon_2_files/output_122_0.png)
    


We see that the expected net benefit of our model remains uniformly better than the treat-all and treat-none policies up to about 60% threshold.

## Mean and Most-Frequent Imputation <a class="anchor" id="mean-imputation"></a>

The complete case analysis is a straightforward but wasteful approach to handling missing data. Mean imputation is another quick fix for missing data, replacing missing values with their mean (or the most frequent category for categorical data). We should, however, keep in mind that mean imputation can distort relationships among variables (biasing regression estimates) [[2](#2)]. 

Let us first check which variables are missing in the data.


```python
framingham.isna().sum(axis = 0)
```




    Sex         0
    Age         0
    Edu       105
    Smoker      0
    Cig        29
    Meds       53
    Stroke      0
    Hyp         0
    Diab        0
    Chol       50
    SysP        0
    DiaP        0
    BMI        19
    Hrate       1
    Gluc      388
    TCHD        0
    dtype: int64



First, we impute the categorical variables using the most-frequent imputation.


```python
from sklearn.impute import SimpleImputer

mfreq_imp_fit = SimpleImputer(missing_values =np.nan, strategy="most_frequent").fit(framingham[['Edu','Meds']])
```

We impute **Hrate**, **BMI**, **Gluc**, and **Chol** using their population mean values.


```python
mean_imp_fit = SimpleImputer(missing_values = np.nan, strategy='mean').fit(framingham[['Chol','BMI','Hrate','Gluc']])
```

 As far as **Cigs** is concerned, we can do a bit better. A nonsmoker would smoke zero cigarettes per day. However, there is no nonsmoker in the data with **Cigs** missing.


```python
((framingham['Cig'].isna()) & (framingham['Smoker'] == 0)).any()
```




    False



Hence, we impute missing **Cigs** values with the conditional mean for smokers. 


```python
mean_imp_fit_cig = SimpleImputer(missing_values = np.nan, strategy='mean').fit(pd.DataFrame(framingham.loc[framingham['Smoker'] == 1]['Cig']))
```

Let us fit the model for the imputed dataset.


```python
framingham_mean_imp = framingham.copy()
framingham_mean_imp[['Chol','BMI','Hrate','Gluc']] = \
    mean_imp_fit.transform(framingham_mean_imp[['Chol','BMI','Hrate','Gluc']])
framingham_mean_imp[['Cig']] = mean_imp_fit_cig.transform(framingham_mean_imp[['Cig']])
framingham_mean_imp[['Edu','Meds']] = mfreq_imp_fit.transform(framingham_mean_imp[['Edu','Meds']])

lr_full_mean_imp = smf.logit(formula='TCHD ~ Sex + bs(Age, df=3) + C(Edu, Poly) + bs(Cig, df=3)  + Meds + Stroke + Hyp + Diab + bs(Chol, df=3) + \
                     bs(SysP, df=3) + bs(DiaP, df=3) + bs(BMI, df=3) + bs(Hrate, df=3) + bs(Gluc, df=3) + \
                     Age:(Cig + Stroke + Hyp + Diab + Chol + SysP + DiaP + BMI + Hrate + Gluc) + \
                     Sex:(Cig + Stroke + Hyp + Diab + Chol + SysP + DiaP + BMI + Hrate + Gluc)', data=framingham_mean_imp)                            

lr_full_mean_imp_fit = lr_full_mean_imp.fit(disp=0)
print(lr_full_mean_imp_fit.summary())

```

                               Logit Regression Results                           
    ==============================================================================
    Dep. Variable:                   TCHD   No. Observations:                 4238
    Model:                          Logit   Df Residuals:                     4185
    Method:                           MLE   Df Model:                           52
    Date:                Thu, 04 Dec 2025   Pseudo R-squ.:                  0.1268
    Time:                        19:57:53   Log-Likelihood:                -1576.8
    converged:                       True   LL-Null:                       -1805.8
    Covariance Type:            nonrobust   LLR p-value:                 2.556e-66
    ==========================================================================================
                                 coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------------------
    Intercept                 -1.8263      1.400     -1.305      0.192      -4.569       0.917
    C(Edu, Poly).Linear        0.0682      0.109      0.624      0.533      -0.146       0.283
    C(Edu, Poly).Quadratic     0.1750      0.110      1.592      0.111      -0.040       0.390
    C(Edu, Poly).Cubic        -0.0195      0.109     -0.179      0.858      -0.234       0.195
    Sex                       -2.8724      1.202     -2.389      0.017      -5.229      -0.516
    bs(Age, df=3)[0]           3.1452      1.436      2.191      0.028       0.332       5.959
    bs(Age, df=3)[1]           3.8273      1.893      2.021      0.043       0.116       7.538
    bs(Age, df=3)[2]           5.1811      2.783      1.862      0.063      -0.273      10.635
    bs(Cig, df=3)[0]           0.3986      0.860      0.464      0.643      -1.286       2.083
    bs(Cig, df=3)[1]           2.6841      1.545      1.738      0.082      -0.344       5.712
    bs(Cig, df=3)[2]          -0.4488      2.618     -0.171      0.864      -5.581       4.683
    Meds                       0.2307      0.226      1.019      0.308      -0.213       0.674
    Stroke                    -6.8548      6.148     -1.115      0.265     -18.905       5.195
    Hyp                       -0.3234      0.997     -0.324      0.746      -2.277       1.630
    Diab                       4.2167      2.376      1.774      0.076      -0.441       8.874
    bs(Chol, df=3)[0]         -1.1082      2.021     -0.548      0.583      -5.069       2.852
    bs(Chol, df=3)[1]          7.3042      3.258      2.242      0.025       0.918      13.690
    bs(Chol, df=3)[2]          4.4244      4.763      0.929      0.353      -4.911      13.759
    bs(SysP, df=3)[0]          1.2083      2.409      0.502      0.616      -3.513       5.930
    bs(SysP, df=3)[1]         -4.2466      4.578     -0.928      0.354     -13.220       4.727
    bs(SysP, df=3)[2]          2.9050      7.552      0.385      0.700     -11.896      17.706
    bs(DiaP, df=3)[0]         -1.8622      1.963     -0.949      0.343      -5.709       1.985
    bs(DiaP, df=3)[1]          4.1843      2.882      1.452      0.147      -1.465       9.834
    bs(DiaP, df=3)[2]          4.3162      4.409      0.979      0.328      -4.325      12.957
    bs(BMI, df=3)[0]          -1.0413      1.551     -0.671      0.502      -4.081       1.999
    bs(BMI, df=3)[1]           1.1008      2.459      0.448      0.654      -3.718       5.920
    bs(BMI, df=3)[2]           0.6143      3.854      0.159      0.873      -6.939       8.167
    bs(Hrate, df=3)[0]         0.2406      1.601      0.150      0.881      -2.897       3.379
    bs(Hrate, df=3)[1]        -1.5893      1.971     -0.806      0.420      -5.452       2.274
    bs(Hrate, df=3)[2]        -1.1203      3.368     -0.333      0.739      -7.722       5.481
    bs(Gluc, df=3)[0]         -4.7352      2.415     -1.960      0.050      -9.469      -0.001
    bs(Gluc, df=3)[1]         -2.9557      4.017     -0.736      0.462     -10.830       4.918
    bs(Gluc, df=3)[2]        -10.5653      6.772     -1.560      0.119     -23.839       2.708
    Age:Cig                   -0.0002      0.001     -0.360      0.719      -0.001       0.001
    Age:Stroke                 0.1297      0.105      1.233      0.217      -0.076       0.336
    Age:Hyp                    0.0159      0.018      0.892      0.372      -0.019       0.051
    Age:Diab                  -0.0782      0.042     -1.873      0.061      -0.160       0.004
    Age:Chol                  -0.0002      0.000     -1.507      0.132      -0.000    6.15e-05
    Age:SysP                   0.0003      0.001      0.669      0.504      -0.001       0.001
    Age:DiaP                  -0.0013      0.001     -1.700      0.089      -0.003       0.000
    Age:BMI                   -0.0003      0.001     -0.216      0.829      -0.003       0.002
    Age:Hrate                  0.0003      0.000      0.526      0.599      -0.001       0.001
    Age:Gluc                   0.0006      0.000      1.777      0.076    -5.8e-05       0.001
    Sex:Cig                    0.0021      0.009      0.223      0.824      -0.016       0.020
    Sex:Stroke                 0.5421      1.033      0.525      0.600      -1.482       2.566
    Sex:Hyp                   -0.5801      0.272     -2.131      0.033      -1.114      -0.047
    Sex:Diab                   0.1846      0.611      0.302      0.763      -1.014       1.383
    Sex:Chol                   0.0033      0.002      1.555      0.120      -0.001       0.008
    Sex:SysP                   0.0146      0.007      1.942      0.052      -0.000       0.029
    Sex:DiaP                   0.0028      0.012      0.234      0.815      -0.021       0.026
    Sex:BMI                 9.847e-05      0.026      0.004      0.997      -0.050       0.050
    Sex:Hrate                  0.0051      0.008      0.619      0.536      -0.011       0.021
    Sex:Gluc                   0.0014      0.004      0.326      0.744      -0.007       0.010
    ==========================================================================================
    

We could repeat the inference for the imputed data using the same methods as for the complete-case analysis. However, such an inference would 
not account for the fact that these data were imputed. To obtain inferences that account for imputation, we can use a pairs bootstrap. The confidence intervals can be computed using the percentile-based method, as we did in Part One for our CRE model. The significance of predictors can be tested using the *bootstrap Wald test*.

First, we compute the Wald test statistic for the original sample: $W = \hat{\theta}^T\, \mathrm{Cov}(\hat{\theta})^{-1} \hat{\theta}$. Then, we compute for each bootstrap sample the Wald test statistic $W' = (\theta' - \hat{\theta})^T\, \mathrm{Cov}(\theta')^{-1} (\theta' - \hat{\theta})$, where $\theta'$ is the estimate for the bootstrap sample. The p-value for the test is then $\frac{\mathrm{count}\; W' > W}{\mathrm{count}\; W'}$. The idea behind the test is that under the alternative, the Wald test statistic $W$ between $\hat{\theta}$ and the null $\theta_0 = 0$ should be much larger than the Wald test statistic $W'$ between $\hat{\theta}$ and $\theta'$ [[11](#11)].

To demonstrate the bootstrap Wald test, we will perform it to test the significance of interactions, **stroke**, and **age**.


```python
# wald statistic for interactions
params = lr_full_mean_imp_fit.params.iloc[range(33,53)]
cov_params = lr_full_mean_imp_fit.cov_params().iloc[range(33,53)].iloc[:,range(33,53)]
wald_stat = np.transpose(params) @ np.linalg.inv(cov_params) @ params

# wald statistic for stroke
params2 = lr_full_mean_imp_fit.params.iloc[[12,34,44]]
cov_params2 = lr_full_mean_imp_fit.cov_params().iloc[[12,34,44]].iloc[:,[12,34,44]]
wald_stat2 = np.transpose(params2) @ np.linalg.inv(cov_params2) @ params2

# wald statistic for age
params3 = lr_full_mean_imp_fit.params.iloc[[*range(5,8),*range(33,43)]]
cov_params3 = lr_full_mean_imp_fit.cov_params().iloc[[*range(5,8),*range(33,43)]].iloc[:,[*range(5,8),*range(33,43)]]
wald_stat3 = np.transpose(params3) @ np.linalg.inv(cov_params3) @ params3
```


```python
np.random.seed(123)
nb = 1000
wald_test = pd.DataFrame(index=range(nb),columns = ['Wald'])
wald_test2 = pd.DataFrame(index=range(nb),columns = ['Wald'])
wald_test3 = pd.DataFrame(index=range(nb),columns = ['Wald'])

import warnings
warnings.filterwarnings('ignore')

for i in range(nb):
    rand_ind = np.random.choice(range(len(framingham)), size=len(framingham), replace=True)
    framingham_new = framingham.iloc[rand_ind]
    
    mean_imp_fit_cig_new = SimpleImputer(missing_values = np.nan, strategy='mean').fit(pd.DataFrame(framingham_new.loc[framingham_new['Smoker'] == 1]['Cig']))
    mean_imp_fit_new = SimpleImputer(missing_values = np.nan, strategy='mean').fit(framingham_new[['Chol','BMI','Hrate','Gluc']])
    mfreq_imp_fit_new = SimpleImputer(missing_values =np.nan, strategy="most_frequent").fit(framingham_new[['Edu','Meds']])
    
    framingham_mean_imp_new = framingham_new.copy()
    framingham_mean_imp_new[['Chol','BMI','Hrate','Gluc']] = \
        mean_imp_fit.transform(framingham_mean_imp_new[['Chol','BMI','Hrate','Gluc']])
    framingham_mean_imp_new[['Cig']] = mean_imp_fit_cig.transform(framingham_mean_imp_new[['Cig']])
    framingham_mean_imp_new[['Edu','Meds']] = mfreq_imp_fit.transform(framingham_mean_imp_new[['Edu','Meds']])
    
    lr_full_mean_imp_new = smf.logit(formula='TCHD ~ Sex + bs(Age, df=3) + C(Edu, Poly) + bs(Cig, df=3) + Meds + Stroke + Hyp + Diab + bs(Chol, df=3) + \
                         bs(SysP, df=3) + bs(DiaP, df=3) + bs(BMI, df=3) + bs(Hrate, df=3) + bs(Gluc, df=3) + \
                         Age:(Cig + Stroke + Hyp + Diab + Chol + SysP + DiaP + BMI + Hrate + Gluc) + \
                         Sex:(Cig + Stroke + Hyp + Diab + Chol + SysP + DiaP + BMI + Hrate + Gluc)', data=framingham_mean_imp_new)
     
    try:
        lr_full_mean_imp_new_fit = lr_full_mean_imp_new.fit(disp=0)                     
         
        params_new = (lr_full_mean_imp_new_fit.params.iloc[range(33,53)] - params)
        cov_params_new  = lr_full_mean_imp_new_fit.cov_params().iloc[range(33,53)].iloc[:,range(33,53)]
        wald_test.iloc[i] = np.transpose(params_new) @ np.linalg.inv(cov_params_new) @ params_new
        
        params_new2 = (lr_full_mean_imp_new_fit.params.iloc[[12,34,44]] - params2)
        cov_params_new2  = lr_full_mean_imp_new_fit.cov_params().iloc[[12,34,44]].iloc[:,[12,34,44]]
        wald_test2.iloc[i] = np.transpose(params_new2) @ np.linalg.inv(cov_params_new2) @ params_new2
        
        params_new3 = (lr_full_mean_imp_new_fit.params.iloc[[*range(5,8),*range(33,43)]] - params3)
        cov_params_new3  = lr_full_mean_imp_new_fit.cov_params().iloc[[*range(5,8),*range(33,43)]].iloc[:,[*range(5,8),*range(33,43)]]
        wald_test3.iloc[i] = np.transpose(params_new3) @ np.linalg.inv(cov_params_new3) @ params_new3
        
    except:
        wald_test.iloc[i] = np.nan
        wald_test2.iloc[i] = np.nan
        wald_test3.iloc[i] = np.nan
    
warnings.filterwarnings('default')
```


```python
(wald_stat <  wald_test).mean() # p-value for interactions
```




    Wald    0.262
    dtype: float64




```python
(wald_stat2 < wald_test2).mean() # p-value for stroke
```




    Wald    0.094
    dtype: float64




```python
(wald_stat3 < wald_test3).mean() # p-value for age
```




    Wald    0.001
    dtype: float64



We observe that interactions are not significant, and **age** is highly significant as was the case in the complete case analysis. Interestingly enough, the **stroke** that was largely nonsignificant in the complete case analysis (due to the lack of data) is now borderline significant. This demonstrates that imputation can be pretty helpful. However, we still need to keep in mind that the mean/most-frequent imputation provides biased estimates even under the MCAR (missing completely at random) condition, and the bootstrap does not account for this. Consequently, we should prefer other imputation methods if precise inference is our primary concern. 

Let us investigate the predictive performance of our model with the mean/most-frequent imputation.


```python
np.random.seed(123)

rep = 100
folds = 10
kf = KFold(n_splits=10) # create folds

metrics_cv_mean_imp =  pd.DataFrame(index=range(rep*folds),columns = ['c-index (AUC)','Brier score','log score','calibration'])
dca_curve_cv_mean_imp  = pd.DataFrame(index=range(rep*folds),columns = range(100))

k = 0
for i in range(rep):
    idx_cv = np.random.choice([*range(len(framingham))],len(framingham), replace=False)
    
    for j, (train_index, test_index) in enumerate(kf.split(idx_cv)):
        
        train_set = idx_cv[train_index]
        test_set = idx_cv[test_index]
        
        framingham_train = framingham.iloc[train_set]
        framingham_test = framingham.iloc[test_set]
        
        mean_imp_fit_cig_new = SimpleImputer(missing_values = np.nan, strategy='mean').fit(pd.DataFrame(framingham_train.loc[framingham_train['Smoker'] == 1]['Cig']))
        mean_imp_fit_new = SimpleImputer(missing_values = np.nan, strategy='mean').fit(framingham_train[['Chol','BMI','Hrate','Gluc']])
        mfreq_imp_fit_new = SimpleImputer(missing_values =np.nan, strategy="most_frequent").fit(framingham_train[['Edu','Meds']])

        framingham_train_mean_imp = framingham_train.copy()
        framingham_train_mean_imp[['Cig']] = mean_imp_fit_cig_new.transform(framingham_train_mean_imp[['Cig']])
        framingham_train_mean_imp[['Chol','BMI','Hrate','Gluc']] = \
            mean_imp_fit_new.transform(framingham_train_mean_imp[['Chol','BMI','Hrate','Gluc']])
        framingham_train_mean_imp[['Edu','Meds']] = mfreq_imp_fit_new.transform(framingham_train_mean_imp[['Edu','Meds']])
            

        lr_cv_new = smf.logit(formula='TCHD ~ Sex + bs(Age, lower_bound=30, upper_bound = 75, df=3) + C(Edu, Poly) + \
                             bs(Cig, lower_bound= 0, upper_bound = 80, df=3) + Meds + Stroke + Hyp + Diab + bs(Chol, lower_bound= 100, upper_bound = 700, df=3) + \
                             bs(SysP, lower_bound= 80, upper_bound = 300, df=3) + bs(DiaP, lower_bound= 45, upper_bound = 150, df=3) + \
                             bs(BMI, lower_bound= 15, upper_bound = 60, df=3) + bs(Hrate, lower_bound= 40, upper_bound = 150, df=3) + \
                             bs(Gluc, df=3, lower_bound= 40, upper_bound = 400)',
                             data=framingham_train_mean_imp)
            
        framingham_test_mean_imp = framingham_test.copy()
        framingham_test_mean_imp[['Cig']] = mean_imp_fit_cig_new.transform(framingham_test_mean_imp[['Cig']])
        framingham_test_mean_imp[['Chol','BMI','Hrate','Gluc']] = \
            mean_imp_fit_new.transform(framingham_test_mean_imp[['Chol','BMI','Hrate','Gluc']])
        framingham_test_mean_imp[['Edu','Meds']] = mfreq_imp_fit_new.transform(framingham_test_mean_imp[['Edu','Meds']])
            
            
        y_pred_prob_new = lr_cv_new.fit(disp=0).predict(framingham_test_mean_imp) 
        y_obs_new = framingham_test_mean_imp['TCHD']
        logitp_new = np.log(y_pred_prob_new/(1-y_pred_prob_new))
        
        metrics_cv_mean_imp.iloc[k,0] = roc_auc_score(y_obs_new,y_pred_prob_new)
        metrics_cv_mean_imp.iloc[k,1] = brier_score_loss(y_obs_new,y_pred_prob_new)
        metrics_cv_mean_imp.iloc[k,2] = log_loss(y_obs_new,y_pred_prob_new)
        metrics_cv_mean_imp.iloc[k,3] = sm.Logit(endog= y_obs_new,exog = pd.DataFrame(logitp_new).assign(const=1)).fit(disp=0).params.iloc[0]
        
        for n in range(100):
            conf_table_new = confusion_matrix(y_obs_new,(y_pred_prob_new > dca_thresholds[n]).astype(int))
            dca_curve_cv_mean_imp.iloc[k,n] = 1/len(test_set)*\
                (conf_table_new[1,1] - conf_table_new[0,1]*dca_thresholds[n]/(1-dca_thresholds[n]))
            
        k = k + 1
```


```python
metrics_cv_mean_imp.mean()
```




    c-index (AUC)     0.72299
    Brier score      0.117186
    log score        0.387034
    calibration      0.874847
    dtype: object




```python
plt.plot(dca_thresholds,dca_curve_cv_mean_imp.mean(),color = 'blue')
plt.plot(dca_thresholds,tnone_curve, color = 'red')
plt.plot(dca_thresholds,tall_curve, color = 'green')
plt.ylim(dca_curve[99]-0.01, dca_curve[0]+0.01);
```


    
![png](First_circle_prevon_2_files/output_145_0.png)
    


We observe that the results are fairly similar to the complete case analysis.

## k-NN Imputation <a class="anchor" id="knn"></a>

Imputation based on the k-NN algorithm is another popular and well-performing imputation technique [[12](#12),[13](#13)]. Unlike the mean/most-frequent imputation, it preserves the relations between variables. Apart from computational complexity for very large datasets, its main disadvantage is that we have to retain the entire original dataset to impute new values.


```python
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

variables = ['Sex','Age','Edu','Cig','Meds','Stroke','Hyp','Diab','Chol','SysP','DiaP','BMI','Hrate','Gluc','Smoker']
predictors = framingham[variables]

scaler = StandardScaler().fit(predictors) 
predictors_scaled = scaler.transform(predictors)
knn_imputer = KNNImputer(n_neighbors=10).fit(predictors_scaled)

framingham_knn_imp = framingham.copy()
# rescale the predictors so that the Euclidean distances are in a comparable scale in all dimensions
framingham_knn_imp[variables] = scaler.inverse_transform(knn_imputer.transform(predictors_scaled))
# kNN averages the values of predictors of neighbouring observations -> rounding to obtain values for categorical predictors
framingham_knn_imp['Edu'] = round(framingham_knn_imp['Edu'])
framingham_knn_imp['Meds'] = round(framingham_knn_imp['Meds'])

lr_full_knn_imp = smf.logit(formula='TCHD ~ Sex + bs(Age, df=3) + C(Edu, Poly) + bs(Cig, df=3)  + Meds + Stroke + Hyp + Diab + bs(Chol, df=3) + \
                     bs(SysP, df=3) + bs(DiaP, df=3) + bs(BMI, df=3) + bs(Hrate, df=3) + bs(Gluc, df=3) + \
                     Age:(Cig + Stroke + Hyp + Diab + Chol + SysP + DiaP + BMI + Hrate + Gluc) + \
                     Sex:(Cig + Stroke + Hyp + Diab + Chol + SysP + DiaP + BMI + Hrate + Gluc)', data=framingham_knn_imp)                            

lr_full_knn_imp_fit = lr_full_knn_imp.fit(disp=0)
print(lr_full_knn_imp_fit.summary())

```

                               Logit Regression Results                           
    ==============================================================================
    Dep. Variable:                   TCHD   No. Observations:                 4238
    Model:                          Logit   Df Residuals:                     4185
    Method:                           MLE   Df Model:                           52
    Date:                Fri, 05 Dec 2025   Pseudo R-squ.:                  0.1271
    Time:                        10:55:30   Log-Likelihood:                -1576.2
    converged:                       True   LL-Null:                       -1805.8
    Covariance Type:            nonrobust   LLR p-value:                 1.492e-66
    ==========================================================================================
                                 coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------------------
    Intercept                 -1.8561      1.401     -1.325      0.185      -4.601       0.889
    C(Edu, Poly).Linear        0.0702      0.110      0.639      0.523      -0.145       0.285
    C(Edu, Poly).Quadratic     0.1590      0.109      1.459      0.144      -0.055       0.373
    C(Edu, Poly).Cubic        -0.0079      0.108     -0.074      0.941      -0.219       0.203
    Sex                       -2.7393      1.204     -2.275      0.023      -5.100      -0.379
    bs(Age, df=3)[0]           3.2362      1.436      2.253      0.024       0.421       6.051
    bs(Age, df=3)[1]           4.0905      1.898      2.156      0.031       0.371       7.810
    bs(Age, df=3)[2]           5.5320      2.787      1.985      0.047       0.069      10.995
    bs(Cig, df=3)[0]           0.3575      0.860      0.416      0.678      -1.328       2.043
    bs(Cig, df=3)[1]           2.7432      1.544      1.777      0.076      -0.282       5.769
    bs(Cig, df=3)[2]          -0.5234      2.625     -0.199      0.842      -5.669       4.622
    Meds                       0.2373      0.227      1.047      0.295      -0.207       0.682
    Stroke                    -6.9505      6.155     -1.129      0.259     -19.015       5.114
    Hyp                       -0.3130      0.998     -0.314      0.754      -2.268       1.642
    Diab                       3.0253      2.537      1.193      0.233      -1.947       7.997
    bs(Chol, df=3)[0]         -1.0271      2.023     -0.508      0.612      -4.992       2.938
    bs(Chol, df=3)[1]          7.3336      3.256      2.252      0.024       0.952      13.715
    bs(Chol, df=3)[2]          4.7672      4.764      1.001      0.317      -4.570      14.105
    bs(SysP, df=3)[0]          0.9782      2.411      0.406      0.685      -3.748       5.705
    bs(SysP, df=3)[1]         -4.5580      4.579     -0.995      0.320     -13.533       4.417
    bs(SysP, df=3)[2]          2.1868      7.559      0.289      0.772     -12.628      17.001
    bs(DiaP, df=3)[0]         -1.7006      1.969     -0.864      0.388      -5.559       2.158
    bs(DiaP, df=3)[1]          4.3834      2.887      1.518      0.129      -1.275      10.042
    bs(DiaP, df=3)[2]          4.6778      4.421      1.058      0.290      -3.988      13.343
    bs(BMI, df=3)[0]          -1.1877      1.551     -0.766      0.444      -4.228       1.853
    bs(BMI, df=3)[1]           1.0430      2.464      0.423      0.672      -3.787       5.873
    bs(BMI, df=3)[2]           0.3403      3.866      0.088      0.930      -7.237       7.917
    bs(Hrate, df=3)[0]         0.2415      1.602      0.151      0.880      -2.899       3.382
    bs(Hrate, df=3)[1]        -1.6103      1.971     -0.817      0.414      -5.473       2.252
    bs(Hrate, df=3)[2]        -1.1412      3.369     -0.339      0.735      -7.744       5.462
    bs(Gluc, df=3)[0]         -3.6242      2.417     -1.499      0.134      -8.362       1.113
    bs(Gluc, df=3)[1]         -0.3958      4.131     -0.096      0.924      -8.492       7.700
    bs(Gluc, df=3)[2]         -7.1874      6.819     -1.054      0.292     -20.553       6.179
    Age:Cig                   -0.0002      0.001     -0.328      0.743      -0.001       0.001
    Age:Stroke                 0.1312      0.105      1.247      0.213      -0.075       0.337
    Age:Hyp                    0.0156      0.018      0.875      0.382      -0.019       0.050
    Age:Diab                  -0.0602      0.044     -1.372      0.170      -0.146       0.026
    Age:Chol                  -0.0002      0.000     -1.550      0.121      -0.000    5.57e-05
    Age:SysP                   0.0004      0.001      0.753      0.451      -0.001       0.001
    Age:DiaP                  -0.0014      0.001     -1.762      0.078      -0.003       0.000
    Age:BMI                   -0.0002      0.001     -0.156      0.876      -0.003       0.003
    Age:Hrate                  0.0003      0.000      0.531      0.596      -0.001       0.001
    Age:Gluc                   0.0004      0.000      1.289      0.197      -0.000       0.001
    Sex:Cig                    0.0014      0.009      0.154      0.878      -0.017       0.019
    Sex:Stroke                 0.5366      1.033      0.520      0.603      -1.488       2.561
    Sex:Hyp                   -0.5719      0.272     -2.100      0.036      -1.106      -0.038
    Sex:Diab                   0.3564      0.634      0.562      0.574      -0.886       1.599
    Sex:Chol                   0.0034      0.002      1.566      0.117      -0.001       0.008
    Sex:SysP                   0.0148      0.007      1.969      0.049    6.63e-05       0.029
    Sex:DiaP                   0.0022      0.012      0.181      0.856      -0.021       0.026
    Sex:BMI                    0.0001      0.026      0.006      0.995      -0.050       0.050
    Sex:Hrate                  0.0053      0.008      0.645      0.519      -0.011       0.021
    Sex:Gluc                  -0.0002      0.004     -0.038      0.970      -0.009       0.009
    ==========================================================================================
    

Let us compare the performance of our model with the k-NN imputation with the mean/most-frequent imputation.


```python
# cross-validation
np.random.seed(123)

rep = 100
folds = 10
kf = KFold(n_splits=10) # create folds

metrics_cv_knn_imp =  pd.DataFrame(index=range(rep*folds),columns = ['c-index (AUC)','Brier score','log score','calibration'])
dca_curve_cv_knn_imp  = pd.DataFrame(index=range(rep*folds),columns = range(100))

k = 0
for i in range(rep):
    idx_cv = np.random.choice([*range(len(framingham))],len(framingham), replace=False)
    
    for j, (train_index, test_index) in enumerate(kf.split(idx_cv)):
        
        train_set = idx_cv[train_index]
        test_set = idx_cv[test_index]
        
        framingham_train = framingham.iloc[train_set]
        framingham_test = framingham.iloc[test_set]
        
        predictors_new = framingham_train[variables]
        scaler_new = StandardScaler().fit(predictors_new)
        predictors_scaled_new = scaler_new.transform(predictors_new)
        knn_imputer_new = KNNImputer(n_neighbors=10).fit(predictors_scaled_new)
        
        
        framingham_knn_new = framingham_train.copy()
        framingham_knn_new[variables] = scaler_new.inverse_transform(knn_imputer_new.transform(predictors_scaled_new))
        framingham_knn_new['Edu'] = round(framingham_knn_new['Edu'])
        framingham_knn_new['Meds'] = round(framingham_knn_new['Meds'])
        
        
        lr_cv_new = smf.logit(formula='TCHD ~ Sex + bs(Age, lower_bound=30, upper_bound = 75, df=3) + C(Edu, Poly) + \
                             bs(Cig, lower_bound= 0, upper_bound = 80, df=3) + Meds + Stroke + Hyp + Diab + bs(Chol, lower_bound= 100, upper_bound = 700, df=3) + \
                             bs(SysP, lower_bound= 80, upper_bound = 300, df=3) + bs(DiaP, lower_bound= 45, upper_bound = 150, df=3) + \
                             bs(BMI, lower_bound= 15, upper_bound = 60, df=3) + bs(Hrate, lower_bound= 40, upper_bound = 150, df=3) + \
                             bs(Gluc, df=3, lower_bound= 30, upper_bound = 400)',
                             data=framingham_knn_new)
        
        predictors_test =  framingham_test[variables]
        predictors_test_scaled = scaler_new.transform(predictors_test)
        framingham_knn_test = framingham_test.copy()
        framingham_knn_test[variables] = scaler_new.inverse_transform(knn_imputer_new.transform(predictors_test_scaled))
        framingham_knn_test['Edu'] = round(framingham_knn_test['Edu'])
        framingham_knn_test['Meds'] = round(framingham_knn_test['Meds'])
        
        
      
        y_pred_prob_new = lr_cv_new.fit(disp=0).predict(framingham_knn_test) 
        y_obs_new = framingham_knn_test['TCHD']
        logitp_new = np.log(y_pred_prob_new/(1-y_pred_prob_new))
        
        metrics_cv_knn_imp.iloc[k,0] = roc_auc_score(y_obs_new,y_pred_prob_new)
        metrics_cv_knn_imp.iloc[k,1] = brier_score_loss(y_obs_new,y_pred_prob_new)
        metrics_cv_knn_imp.iloc[k,2] = log_loss(y_obs_new,y_pred_prob_new)
        metrics_cv_knn_imp.iloc[k,3] = sm.Logit(endog= y_obs_new,exog = pd.DataFrame(logitp_new).assign(const=1)).fit(disp=0).params.iloc[0]
        
        for n in range(100):
            conf_table_new = confusion_matrix(y_obs_new,(y_pred_prob_new > dca_thresholds[n]).astype(int))
            dca_curve_cv_knn_imp.iloc[k,n] = 1/len(test_set)*\
                (conf_table_new[1,1] - conf_table_new[0,1]*dca_thresholds[n]/(1-dca_thresholds[n]))
            
        k = k + 1
```


```python
metrics_cv_knn_imp.mean()
```




    c-index (AUC)    0.723612
    Brier score      0.117111
    log score        0.386755
    calibration      0.875124
    dtype: object




```python
plt.plot(dca_thresholds,dca_curve_cv_knn_imp.mean(),color = 'blue')
plt.plot(dca_thresholds,tnone_curve, color = 'red')
plt.plot(dca_thresholds,tall_curve, color = 'green')
plt.ylim(dca_curve[99]-0.01, dca_curve[0]+0.01);
```


    
![png](First_circle_prevon_2_files/output_152_0.png)
    


We observe that the results are almost identical to mean/most-frequent imputation. We obtained only a marginal improvement in terms of the performance metrics.

## MissForrest Imputation <a class="anchor" id="missforrest"></a>

MissForrest is an imputation algorithm based on random forests (Python implementation uses lgbm by default) that model the relations between each predictor and the rest of the predictors [[14](#14)], and it is one of the best-performing imputational algorithms by numerous computational experiments [[15](#15),[16](#16),[17](#17)]. However, in comparison to other techniques we used, MissForrest is much more computationally expensive.


```python
from missforest import MissForest

categorical=['Sex','Edu','Smoker','Meds','Stroke','Hyp','Diab']

warnings.filterwarnings('ignore')

MissForest_imputation = MissForest(categorical=categorical)
MissForest_imputation._verbose = 0

MissForest_imputation.fit(x = framingham[variables])
framingham_MissForest = framingham.copy()

framingham_MissForest[variables] =  MissForest_imputation.transform(x = framingham[variables])[variables]

lr_full_missforrest_imp = smf.logit(formula='TCHD ~ Sex + bs(Age, df=3) + C(Edu, Poly) + bs(Cig, df=3)  + Meds + Stroke + Hyp + Diab + bs(Chol, df=3) + \
                     bs(SysP, df=3) + bs(DiaP, df=3) + bs(BMI, df=3) + bs(Hrate, df=3) + bs(Gluc, df=3) + \
                     Age:(Cig + Stroke + Hyp + Diab + Chol + SysP + DiaP + BMI + Hrate + Gluc) + \
                     Sex:(Cig + Stroke + Hyp + Diab + Chol + SysP + DiaP + BMI + Hrate + Gluc)', data=framingham_MissForest)                            

print(lr_full_missforrest_imp.fit(disp=0).summary())
warnings.filterwarnings('default')
```

                               Logit Regression Results                           
    ==============================================================================
    Dep. Variable:                   TCHD   No. Observations:                 4238
    Model:                          Logit   Df Residuals:                     4185
    Method:                           MLE   Df Model:                           52
    Date:                Thu, 04 Dec 2025   Pseudo R-squ.:                  0.1277
    Time:                        23:43:47   Log-Likelihood:                -1575.2
    converged:                       True   LL-Null:                       -1805.8
    Covariance Type:            nonrobust   LLR p-value:                 6.464e-67
    ==========================================================================================
                                 coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------------------
    Intercept                 -1.8211      1.401     -1.300      0.194      -4.567       0.924
    C(Edu, Poly).Linear        0.0687      0.109      0.628      0.530      -0.146       0.283
    C(Edu, Poly).Quadratic     0.1772      0.110      1.612      0.107      -0.038       0.393
    C(Edu, Poly).Cubic        -0.0174      0.109     -0.159      0.874      -0.232       0.197
    Sex                       -2.7252      1.204     -2.263      0.024      -5.086      -0.365
    bs(Age, df=3)[0]           3.1797      1.436      2.214      0.027       0.365       5.995
    bs(Age, df=3)[1]           3.9818      1.898      2.098      0.036       0.262       7.702
    bs(Age, df=3)[2]           5.3632      2.787      1.924      0.054      -0.100      10.827
    bs(Cig, df=3)[0]           0.3647      0.860      0.424      0.672      -1.321       2.050
    bs(Cig, df=3)[1]           2.7485      1.545      1.779      0.075      -0.280       5.777
    bs(Cig, df=3)[2]          -0.5572      2.627     -0.212      0.832      -5.707       4.592
    Meds                       0.2375      0.227      1.047      0.295      -0.207       0.682
    Stroke                    -6.8505      6.155     -1.113      0.266     -18.914       5.213
    Hyp                       -0.3164      0.997     -0.317      0.751      -2.271       1.639
    Diab                       3.1699      2.564      1.237      0.216      -1.855       8.194
    bs(Chol, df=3)[0]         -1.1904      2.020     -0.589      0.556      -5.150       2.769
    bs(Chol, df=3)[1]          7.3948      3.255      2.272      0.023       1.015      13.775
    bs(Chol, df=3)[2]          4.4590      4.762      0.936      0.349      -4.873      13.791
    bs(SysP, df=3)[0]          1.0479      2.413      0.434      0.664      -3.682       5.778
    bs(SysP, df=3)[1]         -4.5124      4.583     -0.985      0.325     -13.494       4.470
    bs(SysP, df=3)[2]          2.3842      7.567      0.315      0.753     -12.446      17.214
    bs(DiaP, df=3)[0]         -1.7018      1.970     -0.864      0.388      -5.564       2.160
    bs(DiaP, df=3)[1]          4.3533      2.888      1.507      0.132      -1.307      10.014
    bs(DiaP, df=3)[2]          4.6495      4.423      1.051      0.293      -4.020      13.319
    bs(BMI, df=3)[0]          -1.2130      1.552     -0.782      0.434      -4.254       1.828
    bs(BMI, df=3)[1]           0.9929      2.465      0.403      0.687      -3.838       5.824
    bs(BMI, df=3)[2]           0.2887      3.865      0.075      0.940      -7.287       7.865
    bs(Hrate, df=3)[0]         0.2750      1.603      0.172      0.864      -2.867       3.417
    bs(Hrate, df=3)[1]        -1.5741      1.971     -0.799      0.425      -5.437       2.289
    bs(Hrate, df=3)[2]        -1.0811      3.371     -0.321      0.748      -7.688       5.526
    bs(Gluc, df=3)[0]         -4.1132      2.422     -1.698      0.089      -8.860       0.633
    bs(Gluc, df=3)[1]         -0.8815      4.134     -0.213      0.831      -8.984       7.221
    bs(Gluc, df=3)[2]         -8.3604      6.831     -1.224      0.221     -21.748       5.028
    Age:Cig                   -0.0002      0.001     -0.321      0.748      -0.001       0.001
    Age:Stroke                 0.1293      0.105      1.229      0.219      -0.077       0.336
    Age:Hyp                    0.0157      0.018      0.885      0.376      -0.019       0.051
    Age:Diab                  -0.0636      0.044     -1.436      0.151      -0.150       0.023
    Age:Chol                  -0.0002      0.000     -1.532      0.125      -0.000    5.81e-05
    Age:SysP                   0.0004      0.001      0.733      0.463      -0.001       0.001
    Age:DiaP                  -0.0014      0.001     -1.753      0.080      -0.003       0.000
    Age:BMI                   -0.0002      0.001     -0.145      0.884      -0.003       0.003
    Age:Hrate                  0.0003      0.000      0.510      0.610      -0.001       0.001
    Age:Gluc                   0.0005      0.000      1.475      0.140      -0.000       0.001
    Sex:Cig                    0.0013      0.009      0.139      0.890      -0.017       0.019
    Sex:Stroke                 0.5525      1.034      0.534      0.593      -1.474       2.579
    Sex:Hyp                   -0.5734      0.272     -2.105      0.035      -1.107      -0.040
    Sex:Diab                   0.4237      0.636      0.666      0.506      -0.824       1.671
    Sex:Chol                   0.0036      0.002      1.664      0.096      -0.001       0.008
    Sex:SysP                   0.0147      0.007      1.955      0.051   -3.49e-05       0.029
    Sex:DiaP                   0.0022      0.012      0.181      0.857      -0.021       0.026
    Sex:BMI                    0.0005      0.026      0.021      0.983      -0.050       0.051
    Sex:Hrate                  0.0053      0.008      0.647      0.518      -0.011       0.021
    Sex:Gluc                  -0.0010      0.004     -0.219      0.827      -0.010       0.008
    ==========================================================================================
    


```python
# cross-validation
np.random.seed(123)

rep = 100
folds = 10
kf = KFold(n_splits=10) # create folds

metrics_cv_missforest_imp =  pd.DataFrame(index=range(rep*folds),columns = ['c-index (AUC)','Brier score','log score','calibration'])
dca_curve_cv_missforest_imp  = pd.DataFrame(index=range(rep*folds),columns = range(100))



warnings.filterwarnings('ignore')

k = 0
for i in range(rep):
    idx_cv = np.random.choice([*range(len(framingham))],len(framingham), replace=False)
    
    for j, (train_index, test_index) in enumerate(kf.split(idx_cv)):
        
        train_set = idx_cv[train_index]
        test_set = idx_cv[test_index]
        
        framingham_train = framingham.iloc[train_set]
        framingham_test = framingham.iloc[test_set]
        
        missforest_imputer_new = MissForest(categorical=categorical)
        missforest_imputer_new._verbose = 0
        missforest_imputer_new_fit = missforest_imputer_new.fit(x = framingham_train[variables])
                
        framingham_missforest_new = framingham_train.copy()
        framingham_missforest_new[variables] = missforest_imputer_new_fit.transform(framingham_train[variables])[variables]
        
        
        lr_cv_new = smf.logit(formula='TCHD ~ Sex + bs(Age, lower_bound=30, upper_bound = 75, df=3) + C(Edu, Poly) + \
                             bs(Cig, lower_bound= 0, upper_bound = 80, df=3) + Meds + Stroke + Hyp + Diab + bs(Chol, lower_bound= 80, upper_bound = 700, df=3) + \
                             bs(SysP, lower_bound= 80, upper_bound = 300, df=3) + bs(DiaP, lower_bound= 45, upper_bound = 150, df=3) + \
                             bs(BMI, lower_bound= 15, upper_bound = 60, df=3) + bs(Hrate, lower_bound= 40, upper_bound = 150, df=3) + \
                             bs(Gluc, df=3, lower_bound= 20, upper_bound = 400)',
                             data = framingham_missforest_new)
        


        framingham_missforest_test = framingham_test.copy()
        framingham_missforest_test[variables] = missforest_imputer_new_fit.transform(framingham_test[variables])[variables]


        y_pred_prob_new = lr_cv_new.fit(disp=0).predict(framingham_missforest_test) 
        y_obs_new = framingham_missforest_test['TCHD']
        logitp_new = np.log(y_pred_prob_new/(1-y_pred_prob_new))
        
        metrics_cv_missforest_imp.iloc[k,0] = roc_auc_score(y_obs_new,y_pred_prob_new)
        metrics_cv_missforest_imp.iloc[k,1] = brier_score_loss(y_obs_new,y_pred_prob_new)
        metrics_cv_missforest_imp.iloc[k,2] = log_loss(y_obs_new,y_pred_prob_new)
        metrics_cv_missforest_imp.iloc[k,3] = sm.Logit(endog= y_obs_new,exog = pd.DataFrame(logitp_new).assign(const=1)).fit(disp=0).params.iloc[0]
        
        for n in range(100):
            conf_table_new = confusion_matrix(y_obs_new,(y_pred_prob_new > dca_thresholds[n]).astype(int))
            dca_curve_cv_missforest_imp.iloc[k,n] = 1/len(test_set)*\
                (conf_table_new[1,1] - conf_table_new[0,1]*dca_thresholds[n]/(1-dca_thresholds[n]))
            
        k = k + 1
```


```python
warnings.filterwarnings('default')
metrics_cv_missforest_imp.mean()
```




    c-index (AUC)    0.724165
    Brier score      0.117025
    log score        0.386542
    calibration      0.874205
    dtype: object




```python
plt.plot(dca_thresholds,dca_curve_cv_missforest_imp.mean(),color = 'blue')
plt.plot(dca_thresholds,tnone_curve, color = 'red')
plt.plot(dca_thresholds,tall_curve, color = 'green')
plt.ylim(dca_curve[99]-0.01, dca_curve[0]+0.01);
```


    
![png](First_circle_prevon_2_files/output_158_0.png)
    


The results are again fairly similar to other imputation techniques, although to be fair, we observe another marginal improvement. 

## Regression Imputation (via chained equation) <a class="anchor" id="mice"></a>

The last imputation we will mention here is the multivariate regression-based imputation. The approach initializes imputation with randomly generated values. Then, the variables are imputed one at a time (using the rest of the data via methods like linear regression/predictive mean matching for continuous variables and logistic regression for binary variables). Once all variables are imputed, the process is iteratively repeated, creating a Markov random process. It is assumed that this process will have a unique stationary distribution (which often seems to be the case for real-world data), i.e., the distribution of our imputed values will not depend on the initial selection of the imputed values if enough iterations are taken [[2](#2)].

Imputation via chained equation is primarily implemented in R in the package *mice*. We described how to set this imputation up in great detail in The Second Circle: Logistic Regression, Part Two (it is quite a bit more involved than MissForrest, because we have to specify all imputation models and we have to make sure that the resulting Markov random process converges to a stationary distribution). 

Lastly, we should note that *mice* is focused chiefly on inference. It can be shown that chained equation-based imputation provides unbiased regression estimates under missing at random (MAR) conditions (i.e., missingness depends on the observed data), provided that the imputation models are correctly specified. The chained equations converge to a unique stationary distribution. There is also a so-called congeniality condition, which is essentially about the compatibility of the imputation model with the models used for later analysis. In addition, imputation is usually performed *multiple times* (i.e., multiple imputed datasets), and the results are pooled together via the so-called Rubin's rules or bootstrap. It can be shown that under correct specification and congeniality, the standard error estimates are also unbiased, resulting in a valid statistical inference [[18](#18)]. 

On the other hand, obtaining predictions is quite a bit clunky (*mice* does not even have a *predict* function). We discuss this in more detail in The Second Circle: Logistic Regression, Part Three.

## References <a class="anchor" id="references"></a>

<a id="1">[1]</a> HARRELL, Frank E., Jr. *Regression Modeling Strategies With Applications to Linear Models, Logistic and Ordinal Regression, and Survival Analysis*. Springer, 2015.

<a id="2">[2]</a> VAN BUUREN, Stef; VAN BUUREN, Stef. *Flexible imputation of missing data*. Boca Raton, FL: CRC press, 2012.

<a id="3">[3]</a> MCEVOY, John William, et al. 2024 ESC Guidelines for the management of elevated blood pressure and hypertension: Developed by the task force on the management of elevated blood pressure and hypertension of the European Society of Cardiology (ESC) and endorsed by the European Society of Endocrinology (ESE) and the European Stroke Organisation (ESO). European heart journal, 2024, 45.38: 3912-4018.

<a id="4">[4]</a> MOOD, Carina. Logistic regression: Why we cannot do what we think we can do, and what we can do about it. *European sociological review*, 2010, 26.1: 67-82.

<a id="5">[5]</a> DUNN, Peter K., et al. Generalized linear models with examples in R. *New York: Springer*, 2018.

<a id="6">[6]</a> VAN HOUWELINGEN, J. C.; LE CESSIE, Saskia. Predictive value of statistical models. *Statistics in medicine*, 1990, 9.11: 1303-1325.

<a id="7">[7]</a> COOK, Nancy R. Use and misuse of the receiver operating characteristic curve in risk prediction. *Circulation*, 2007, 115.7: 928-935.

<a id="8">[8]</a> BURNHAM, Kenneth P.; ANDERSON, David R. Practical use of the information-theoretic approach. In: *Model selection and inference: A practical information-theoretic approach.* New York, NY: Springer New York, 1998. p. 75-117.

<a id="9">[9]</a> HOESSLY, Linard. On misconceptions about the Brier score in binary prediction models. arXiv preprint arXiv:2504.04906, 2025.

<a id="10">[10]</a> VICKERS, Andrew J.; ELKIN, Elena B. Decision curve analysis: a novel method for evaluating prediction models. *Medical Decision Making*, 2006, 26.6: 565-574.

<a id="11">[11]</a> HALL, Peter; WILSON, Susan R. Two guidelines for bootstrap hypothesis testing. Biometrics, 1991, 757-762.

<a id="12">[12]</a> SEU, Kimseth; KANG, Mi-Sun; LEE, HwaMin. An intelligent missing data imputation techniques: A review. *JOIV: International Journal on Informatics Visualization*, 2022, 6.1-2: 278-283.

<a id="13">[13]</a> ALWATEER, Majed, et al. Missing data imputation: A comprehensive review. *Journal of Computer and Communications*, 2024, 12.11: 53-75.

<a id="14">[14]</a>  STEKHOVEN, Daniel J.; BÜHLMANN, Peter. MissForest—non-parametric missing value imputation for mixed-type data. *Bioinformatics*, 2012, 28.1: 112-118.

<a id="15">[15]</a> JADHAV, Anil; PRAMOD, Dhanya; RAMANATHAN, Krishnan. Comparison of performance of data imputation methods for numeric dataset. Applied Artificial Intelligence, 2019, 33.10: 913-933.

<a id="16">[16]</a>  SUN, Yige, et al. Deep learning versus conventional methods for missing data imputation: A review and comparative study. Expert Systems with Applications, 2023, 227: 120201.

<a id="17">[17]</a> JOEL, Luke Oluwaseye; DOORSAMY, Wesley; PAUL, Babu Sena. A comparative study of imputation techniques for missing values in healthcare diagnostic datasets. International Journal of Data Science and Analytics, 2025, 1-17.

<a id="18">[18]</a> MURRAY, Jared S. Multiple imputation: a review of practical and theoretical findings. 2018.
