# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from scipy import stats

np.set_printoptions(legacy='1.25')

# load dataset
life_expectancy = pd.read_csv('C:/Users/elini/Desktop/nine circles 2/Life-Expectancy-Data-Updated.csv')
life_expectancy

# cleaning
# drop redundant column
life_expectancy = life_expectancy.drop('Economy_status_Developing', axis=1)
# turn column into factor and rename
life_expectancy = life_expectancy.rename(columns={'Economy_status_Developed': 'Economy_status'})
life_expectancy['Economy_status'] = life_expectancy['Economy_status'].astype(object)
# rename levels
life_expectancy.loc[life_expectancy['Economy_status'] == 0,'Economy_status'] = 'Developing'
life_expectancy.loc[life_expectancy['Economy_status'] == 1,'Economy_status'] = 'Developed'


# check missing and duplicated rows
life_expectancy.isna().any(axis = None)
life_expectancy.duplicated(keep = False).any(axis = None)

# descrpitive stats
stats.describe(life_expectancy['Infant_deaths'], axis = 0)

# print table
numeric_variables = list(life_expectancy)[3:18]
numeric_basic_stats = np.zeros([len(numeric_variables),6])

for i in range(len(numeric_variables)):
    numeric_basic_stats[i,0] = np.mean(life_expectancy[numeric_variables[i]])
    numeric_basic_stats[i,1] = np.median(life_expectancy[numeric_variables[i]])
    numeric_basic_stats[i,2] = np.min(life_expectancy[numeric_variables[i]])
    numeric_basic_stats[i,3] = np.quantile(life_expectancy[numeric_variables[i]], 0.25)
    numeric_basic_stats[i,4] = np.quantile(life_expectancy[numeric_variables[i]], 0.75)
    numeric_basic_stats[i,5] = np.max(life_expectancy[numeric_variables[i]])

      
pd.DataFrame(data=numeric_basic_stats, index=numeric_variables, columns=['mean','median','min','1st qrt.','3rd qrt.','max'])


# histograms
import matplotlib.pyplot as plt

f,a = plt.subplots(1,3)
a = a.ravel()
for idx,ax in enumerate(a):
    ax.hist(life_expectancy[numeric_variables[idx]], bins=20)
    ax.set_title('Histogram')
    ax.set_xlabel(numeric_variables[idx])
    ax.set_ylabel('Frequency')
plt.tight_layout()


f,a = plt.subplots(1,3)
a = a.ravel()
for idx,ax in enumerate(a):
    ax.hist(life_expectancy[numeric_variables[idx+3]], bins=20)
    ax.set_title('Histogram')
    ax.set_xlabel(numeric_variables[idx+3])
    ax.set_ylabel('Frequency')
plt.tight_layout()

f,a = plt.subplots(1,3)
a = a.ravel()
for idx,ax in enumerate(a):
    ax.hist(life_expectancy[numeric_variables[idx+6]], bins=20)
    ax.set_title('Histogram')
    ax.set_xlabel(numeric_variables[idx+6])
    ax.set_ylabel('Frequency')
plt.tight_layout()

f,a = plt.subplots(1,3)
a = a.ravel()
for idx,ax in enumerate(a):
    ax.hist(life_expectancy[numeric_variables[idx+9]], bins=20)
    ax.set_title('Histogram')
    ax.set_xlabel(numeric_variables[idx+9])
    ax.set_ylabel('Frequency')
plt.tight_layout()

f,a = plt.subplots(1,3)
a = a.ravel()
for idx,ax in enumerate(a):
    ax.hist(life_expectancy[numeric_variables[idx+12]], bins=20)
    ax.set_title('Histogram')
    ax.set_xlabel(numeric_variables[idx+12])
    ax.set_ylabel('Frequency')
plt.tight_layout()


# add logs
life_expectancy['GDP_log'] = np.log(life_expectancy['GDP_per_capita'])
life_expectancy['Pop_log'] = np.log(life_expectancy['Population_mln']+1)

# categorical predictors
life_expectancy['Region'].value_counts()

# VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF

ind = [*range(3,5),*range(6,13),*range(15,18),*range(20,22)]
ind = [list(life_expectancy)[x] for x in ind]

VIF_dataframe = life_expectancy[ind].copy()
VIF_dataframe['Economy_status'] = (life_expectancy['Economy_status'] == 'Developing').astype(int)
VIF_dataframe = VIF_dataframe.assign(const=1)


VIF_vals = [VIF(VIF_dataframe, i) 
        for i in range(0, VIF_dataframe.shape[1])]
VIF_vals = pd.DataFrame({'VIF':VIF_vals},index=VIF_dataframe.columns)
VIF_vals

# we need to consider Under_five_deaths - Infant_deaths

plt.hist((life_expectancy['Under_five_deaths']-life_expectancy['Infant_deaths']), bins=20)
plt.title('Histogram')
plt.xlabel('Under_five_deaths - Infant_deaths')
plt.ylabel('Frequency')

life_expectancy['Child_deaths'] = life_expectancy['Under_five_deaths']-life_expectancy['Infant_deaths']

# InitiaL model
import statsmodels.api as sm
import statsmodels.formula.api as smf


# adult mortality 
adm_fit = smf.ols(formula='Life_expectancy ~ Adult_mortality', data=life_expectancy) 
print(adm_fit.fit().summary())


# pooled model
lm_pooled = smf.ols(formula='Life_expectancy ~ Economy_status + Region + Alcohol_consumption + \
                    Hepatitis_B + Measles + BMI + Polio + Diphtheria + Incidents_HIV + GDP_log + \
                    Pop_log + Thinness_ten_nineteen_years + Thinness_five_nine_years + Schooling + \
                    Infant_deaths + Child_deaths', data=life_expectancy)

print(lm_pooled.fit().summary())                   

# residuals
sm.qqplot(lm_pooled.fit().resid, line='s')

plt.scatter(lm_pooled.fit().fittedvalues, lm_pooled.fit().resid)
plt.title('Residuals vs Predicted')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')


boxplotresid_dataframe = pd.DataFrame({'Resid':lm_pooled.fit().resid, \
                                       'Economy_status':life_expectancy['Economy_status'],
                                       'Region':life_expectancy['Region'],})
boxplotresid_dataframe.boxplot('Resid',by='Economy_status')

boxplotresid_dataframe.boxplot('Resid',by='Region')
plt.xticks([1,2,3,4,5,6,7,8,9], ['Africa','Asia','C.Am.','EU','Md.East','N.Am.','Oceania','nonEU','S.Am.'])


plt.scatter(life_expectancy['GDP_log'],lm_pooled.fit().resid)
plt.title('Residuals vs GDP_log')
plt.xlabel('Log GDP per capita')
plt.ylabel('Residuals')

plt.scatter(life_expectancy['Pop_log'],lm_pooled.fit().resid)
plt.title('Residuals vs Pop_log')
plt.xlabel('Log (Population_mln + 1)')
plt.ylabel('Residuals')


# cook's distance

infl = lm_pooled.fit().get_influence()
sm_fr = infl.summary_frame()
plt.plot(sm_fr['cooks_d'])
plt.xlabel("Cook's distance")
plt.ylabel('Index');

lm_pooled_red = smf.ols(formula='Life_expectancy ~ Economy_status + Region + Alcohol_consumption + \
                    Hepatitis_B + Measles + BMI + Polio + Diphtheria + Incidents_HIV + GDP_log + \
                    Pop_log + Thinness_ten_nineteen_years + Thinness_five_nine_years + Schooling + \
                    Infant_deaths + Child_deaths', data=life_expectancy.loc[sm_fr['cooks_d'] < 0.005])

print(lm_pooled_red.fit().summary()) 

# heteroskedasticity

# HC errors
hc_errors = pd.concat([lm_pooled.fit().bse, lm_pooled.fit(cov_type='HC0').bse, \
                       lm_pooled.fit(cov_type='HC3').bse], axis=1)
hc_errors = hc_errors.rename(columns={0: 'se', 1: 'HC0', 2: 'HC3'})
hc_errors

print(lm_pooled.fit(cov_type='HC3').summary()) 

# pairs bootstrap
np.random.seed(123)
nb = 1000
boot_coef1 = np.zeros([nb,len(lm_pooled.fit().params)])

for i in range(nb):
    rand_ind = np.random.choice(range(len(life_expectancy)), size=len(life_expectancy), replace=True)
    life_expectancy_new = pd.DataFrame(life_expectancy.loc[x] for x in rand_ind)
    lm_pooled_new = smf.ols(formula='Life_expectancy ~ Economy_status + Region + Alcohol_consumption + \
                            Hepatitis_B + Measles + BMI + Polio + Diphtheria + Incidents_HIV + GDP_log + \
                            Pop_log + Thinness_ten_nineteen_years + Thinness_five_nine_years + Schooling + \
                            Infant_deaths + Child_deaths', data=life_expectancy_new)
    boot_coef1[i] = lm_pooled_new.fit().params


pd.DataFrame(boot_coef1, columns=list(lm_pooled_new.fit().params.index)).quantile([0.025,0.975],0)

   
# autocorrelation

plt.scatter(life_expectancy[life_expectancy['Country'] == 'France']['Year'], \
            lm_pooled.fit().resid[life_expectancy['Country'] == 'France'])
plt.title('France')


plt.scatter(life_expectancy[life_expectancy['Country'] == 'Madagascar']['Year'], \
            lm_pooled.fit().resid[life_expectancy['Country'] == 'Madagascar'])
plt.title('Madagascar')



cr_errors = pd.concat([lm_pooled.fit().bse, lm_pooled.fit(cov_type='HC0').bse, \
                       lm_pooled.fit(cov_type='HC3').bse, \
                       lm_pooled.fit(cov_type='cluster',cov_kwds = {'groups':life_expectancy['Country']}).bse], axis=1)
cr_errors = cr_errors.rename(columns={0: 'se', 1: 'HC0', 2: 'HC3', 3: 'CR'})
cr_errors


print(lm_pooled.fit(cov_type='cluster',cov_kwds = {'groups':life_expectancy['Country']}).summary()) 


# pairs cluster bootstrap
np.random.seed(123)
nb = 1000
boot_coef2 = pd.DataFrame(index=range(nb),columns=list(lm_pooled.fit().params.index))

np.zeros([nb,len(lm_pooled.fit().params)])

countries = life_expectancy['Country'].unique()

for i in range(nb):
    rand_ind_country = np.random.choice(range(len(countries)), size=len(life_expectancy), replace=True)
    rand_ind = []
    for j in range(len(countries)):
        rand_ind.append(np.nonzero(life_expectancy['Country'] == countries[rand_ind_country[j]]))
    rand_ind = np.concatenate(rand_ind, axis=1)[0]

    life_expectancy_new = pd.DataFrame(life_expectancy.loc[x] for x in rand_ind)
    lm_pooled_new = smf.ols(formula='Life_expectancy ~ Economy_status + Region + Alcohol_consumption + \
                            Hepatitis_B + Measles + BMI + Polio + Diphtheria + Incidents_HIV + GDP_log + \
                            Pop_log + Thinness_ten_nineteen_years + Thinness_five_nine_years + Schooling + \
                            Infant_deaths + Child_deaths', data=life_expectancy_new)                   
    boot_coef2.loc[i,lm_pooled_new.fit().params.to_frame().index] = lm_pooled_new.fit().params

pd.DataFrame(boot_coef2, columns=list(lm_pooled.fit().params.index)).quantile([0.025,0.975],0).T


# time fixed effects model

life_expectancy_fe = life_expectancy.copy()
life_expectancy_fe['Year_factor'] = life_expectancy_fe['Year'].astype(object)


lm_tfe = smf.ols(formula='Life_expectancy ~ Economy_status + Region + Alcohol_consumption + \
                    Hepatitis_B + Measles + BMI + Polio + Diphtheria + Incidents_HIV + GDP_log + \
                    Pop_log + Thinness_ten_nineteen_years + Thinness_five_nine_years + Schooling + \
                    Infant_deaths + Child_deaths + Year_factor', data=life_expectancy_fe)
                    
print(lm_tfe.fit(cov_type='cluster',cov_kwds = {'groups':life_expectancy_fe['Country']}).summary()) 

hypothesis_year = '(Year_factor[T.2001] = 0,Year_factor[T.2002] = 0,Year_factor[T.2003] = 0,\
Year_factor[T.2004] = 0,Year_factor[T.2005] = 0,Year_factor[T.2006] = 0,Year_factor[T.2007] = 0,\
Year_factor[T.2008] = 0,Year_factor[T.2009] = 0,Year_factor[T.2010] = 0,Year_factor[T.2011] = 0,\
Year_factor[T.2012] = 0,Year_factor[T.2013] = 0,Year_factor[T.2014] = 0,Year_factor[T.2015] = 0)' 
lm_tfe.fit(cov_type='cluster',cov_kwds = {'groups':life_expectancy_fe['Country']}).wald_test(hypothesis_year)
                                 
life_expectancy_fe['pooled_fit'] = lm_pooled.fit().fittedvalues
life_expectancy_fe['tfe_fit'] = lm_tfe.fit().fittedvalues

plt.plot([*range(2000,2016)], life_expectancy_fe[['Year','tfe_fit']].groupby(['Year']).mean(), label="fixed effects")
plt.plot([*range(2000,2016)], life_expectancy_fe[['Year','pooled_fit']].groupby(['Year']).mean(), label="pooled")
plt.xlabel('Year')
plt.ylabel('Predicted Mean Life Expectancy')
plt.legend()

# individual fixed effects model

lm_itfe = smf.ols(formula='Life_expectancy ~  Alcohol_consumption + \
                    Hepatitis_B + Measles + BMI + Polio + Diphtheria + Incidents_HIV + GDP_log + \
                    Pop_log + Thinness_ten_nineteen_years + Thinness_five_nine_years + Schooling + \
                    Infant_deaths + Child_deaths + Year_factor + Country', data=life_expectancy_fe)
                    
                    
print(lm_itfe.fit(cov_type='cluster',cov_kwds = {'groups':life_expectancy_fe['Country']}).summary())      

# random effects model

lm_tre = smf.mixedlm(formula='Life_expectancy ~ Economy_status + Region + Alcohol_consumption + \
                    Hepatitis_B + Measles + BMI + Polio + Diphtheria + Incidents_HIV + GDP_log + \
                    Pop_log + Thinness_ten_nineteen_years + Thinness_five_nine_years + Schooling + \
                    Infant_deaths + Child_deaths + Year_factor', groups=life_expectancy_fe['Country'], \
                    data=life_expectancy_fe)
    
print(lm_tre.fit().summary())
sm.qqplot((pd.DataFrame(lm_tre.fit().random_effects)).iloc[0,:], line='s')

# exogeneity test


alcohol_cent = life_expectancy[['Alcohol_consumption','Country']].groupby('Country').mean().rename(columns={'Alcohol_consumption': 'Alcohol_consumption_cent'})
hepatitis_cent = life_expectancy[['Hepatitis_B','Country']].groupby('Country').mean().rename(columns={'Hepatitis_B': 'Hepatitis_B_cent'})
measles_cent = life_expectancy[['Measles','Country']].groupby('Country').mean().rename(columns={'Measles': 'Measles_cent'})
bmi_cent = life_expectancy[['BMI','Country']].groupby('Country').mean().rename(columns={'BMI': 'BMI_cent'})
polio_cent = life_expectancy[['Polio','Country']].groupby('Country').mean().rename(columns={'Polio': 'Polio_cent'})
diphteria_cent = life_expectancy[['Diphtheria','Country']].groupby('Country').mean().rename(columns={'Diphtheria': 'Diphtheria_cent'})
hiv_cent = life_expectancy[['Incidents_HIV','Country']].groupby('Country').mean().rename(columns={'Incidents_HIV': 'Incidents_HIV_cent'})
gdp_log_cent = life_expectancy[['GDP_log','Country']].groupby('Country').mean().rename(columns={'GDP_log': 'GDP_log_cent'})
pop_log_cent = life_expectancy[['Pop_log','Country']].groupby('Country').mean().rename(columns={'Pop_log': 'Pop_log_cent'})
thinness19_cent = life_expectancy[['Thinness_ten_nineteen_years','Country']].groupby('Country').mean().rename(columns={'Thinness_ten_nineteen_years': 'Thinness_ten_nineteen_years_cent'})
thinness9_cent = life_expectancy[['Thinness_five_nine_years','Country']].groupby('Country').mean().rename(columns={'Thinness_five_nine_years': 'Thinness_five_nine_years_cent'})
schooling_cent = life_expectancy[['Schooling','Country']].groupby('Country').mean().rename(columns={'Schooling': 'Schooling_cent'})
infant_d_cent = life_expectancy[['Infant_deaths','Country']].groupby('Country').mean().rename(columns={'Infant_deaths': 'Infant_deaths_cent'})
child_d_cent = life_expectancy[['Child_deaths','Country']].groupby('Country').mean().rename(columns={'Child_deaths': 'Child_deaths_cent'})


life_expectancy_cent = life_expectancy_fe.merge(alcohol_cent,on = 'Country').merge(hepatitis_cent,on = 'Country').merge(measles_cent,on = 'Country').merge(bmi_cent,on = 'Country')\
.merge(polio_cent,on = 'Country').merge(diphteria_cent,on = 'Country').merge(hiv_cent,on = 'Country').merge(gdp_log_cent,on = 'Country')\
.merge(pop_log_cent,on = 'Country').merge(thinness19_cent,on = 'Country').merge(thinness9_cent,on = 'Country').merge(schooling_cent,on = 'Country')\
.merge(infant_d_cent,on = 'Country').merge(child_d_cent,on = 'Country')

# correlated random effects model

lm_cr = smf.mixedlm(formula='Life_expectancy ~ Economy_status + Region + Alcohol_consumption + \
                    Hepatitis_B + Measles + BMI + Polio + Diphtheria + Incidents_HIV + GDP_log + \
                    Pop_log + Thinness_ten_nineteen_years + Thinness_five_nine_years + Schooling + \
                    Infant_deaths + Child_deaths + Year_factor + Alcohol_consumption_cent + \
                    Hepatitis_B_cent + Measles_cent + BMI_cent + Polio_cent + Diphtheria_cent + \
                    Incidents_HIV_cent + GDP_log_cent + Pop_log_cent + Thinness_ten_nineteen_years_cent + \
                    Thinness_five_nine_years_cent + Schooling_cent + Infant_deaths_cent + Child_deaths_cent', \
                    groups=life_expectancy_cent['Country'], \
                    data=life_expectancy_cent)

print(lm_cr.fit().summary())    

hypothesis_cent = '(Alcohol_consumption_cent = 0,Hepatitis_B_cent = 0, Measles_cent = 0,\
BMI_cent = 0, Polio_cent = 0, Diphtheria_cent = 0, Incidents_HIV_cent = 0,\
GDP_log_cent = 0, Pop_log_cent = 0, Thinness_ten_nineteen_years_cent = 0, Thinness_five_nine_years_cent = 0,\
Schooling_cent = 0, Infant_deaths_cent = 0, Child_deaths_cent = 0)' 
    
lm_cr.fit().wald_test(hypothesis_cent)

# pairs cluster bootstrap
np.random.seed(123)
nb = 1000
boot_coef3 = pd.DataFrame(index=range(nb),columns=list(lm_cr.fit().params.index))

np.zeros([nb,len(lm_cr.fit().params)])

countries = life_expectancy_cent['Country'].unique()

for i in range(nb):
    rand_ind_country = np.random.choice(range(len(countries)), size=len(life_expectancy_cent), replace=True)
    rand_ind = []
    for j in range(len(countries)):
        rand_ind.append(np.nonzero(life_expectancy_cent['Country'] == countries[rand_ind_country[j]]))
    rand_ind = np.concatenate(rand_ind, axis=1)[0]

    life_expectancy_cent_new = pd.DataFrame(life_expectancy_cent.loc[x] for x in rand_ind)
    lm_cr_new = smf.mixedlm(formula='Life_expectancy ~ Economy_status + Region + Alcohol_consumption + \
                    Hepatitis_B + Measles + BMI + Polio + Diphtheria + Incidents_HIV + GDP_log + \
                    Pop_log + Thinness_ten_nineteen_years + Thinness_five_nine_years + Schooling + \
                    Infant_deaths + Child_deaths + Year_factor + Alcohol_consumption_cent + \
                    Hepatitis_B_cent + Measles_cent + BMI_cent + Polio_cent + Diphtheria_cent + \
                    Incidents_HIV_cent + GDP_log_cent + Pop_log_cent + Thinness_ten_nineteen_years_cent + \
                    Thinness_five_nine_years_cent + Schooling_cent + Infant_deaths_cent + Child_deaths_cent', \
                    groups=life_expectancy_cent_new['Country'], \
                    data=life_expectancy_cent_new)                   
    boot_coef3.loc[i,lm_cr_new.fit().params.to_frame().index] = lm_cr_new.fit().params

pd.DataFrame(boot_coef3, columns=list(lm_pooled.fit().params.index)).quantile([0.025,0.975],0).T


# residuals
sm.qqplot(lm_cr.fit().resid, line='s');
plt.title('Residuals Q-Q plot')

plt.hist(lm_cr.fit().resid, bins=100);
plt.title('Histogram')
plt.xlabel('Residuals')
plt.ylabel('Frequency')

# random effects
sm.qqplot((pd.DataFrame(lm_cr.fit().random_effects)).iloc[0,:], line='s');
plt.title('Random effects Q-Q plot')

# residuals vs fitted
plt.scatter(lm_cr.fit().fittedvalues, lm_cr.fit().resid)
plt.title('Residuals vs Predicted')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')

# residuals vs predictors
fig, axs = plt.subplots(1,3)
axs[0].scatter(life_expectancy['Alcohol_consumption'], lm_cr.fit().resid)
axs[0].set_xlabel('Alcohol_consumption')
axs[0].set_ylabel('Residuals')
axs[1].scatter(life_expectancy['Hepatitis_B'], lm_cr.fit().resid)
axs[1].set_xlabel('Hepatitis_B')
axs[2].scatter(life_expectancy['Measles'], lm_cr.fit().resid)
axs[2].set_xlabel('Measles')

fig, axs = plt.subplots(1,3)
axs[0].scatter(life_expectancy['BMI'], lm_cr.fit().resid)
axs[0].set_xlabel('BMI')
axs[0].set_ylabel('Residuals')
axs[1].scatter(life_expectancy['Polio'], lm_cr.fit().resid)
axs[1].set_xlabel('Polio')
axs[2].scatter(life_expectancy['Diphtheria'], lm_cr.fit().resid)
axs[2].set_xlabel('Diphtheria')

fig, axs = plt.subplots(1,3)
axs[0].scatter(life_expectancy['Incidents_HIV'], lm_cr.fit().resid)
axs[0].set_xlabel('Incidents_HIV')
axs[0].set_ylabel('Residuals')
axs[1].scatter(life_expectancy['GDP_log'], lm_cr.fit().resid)
axs[1].set_xlabel('GDP_log')
axs[2].scatter(life_expectancy['Pop_log'], lm_cr.fit().resid)
axs[2].set_xlabel('Pop_log')

fig, axs = plt.subplots(1,3)
axs[0].scatter(life_expectancy['Thinness_ten_nineteen_years'], lm_cr.fit().resid)
axs[0].set_xlabel('Thinness_ten_nineteen')
axs[0].set_ylabel('Residuals')
axs[1].scatter(life_expectancy['Thinness_five_nine_years'], lm_cr.fit().resid)
axs[1].set_xlabel('Thinness_five_nine')
axs[2].scatter(life_expectancy['Schooling'], lm_cr.fit().resid)
axs[2].set_xlabel('Schooling')

fig, axs = plt.subplots(1,2)
axs[0].scatter(life_expectancy['Infant_deaths'], lm_cr.fit().resid)
axs[0].set_xlabel('Infant_deaths')
axs[0].set_ylabel('Residuals')
axs[1].scatter(life_expectancy['Child_deaths'], lm_cr.fit().resid)
axs[1].set_xlabel('Child_deaths')


boxplotresid_dataframe2 = pd.DataFrame({'Resid':lm_cr.fit().resid, \
                                       'Economy_status':life_expectancy['Economy_status'],
                                       'Region':life_expectancy['Region'],})
boxplotresid_dataframe2.boxplot('Resid',by='Economy_status')

boxplotresid_dataframe2.boxplot('Resid',by='Region')
plt.xticks([1,2,3,4,5,6,7,8,9], ['Africa','Asia','C.Am.','EU','Md.East','N.Am.','Oceania','nonEU','S.Am.'])



#  Confidence intervals for predictions

life_expectancy_sort = life_expectancy_cent.sort_values(axis = 0, by = ['Country','Year']).reset_index(drop=True) # sort by Country and Year
life_expectancy_nofrance = life_expectancy_sort.loc[life_expectancy_sort['Country'] != 'France'].reset_index(drop=True)
life_expectancy_france = life_expectancy_sort.loc[life_expectancy_sort['Country'] == 'France'].reset_index(drop=True)


lm_no_france = smf.mixedlm(formula='Life_expectancy ~ Economy_status + Region + Alcohol_consumption + \
                    Hepatitis_B + Measles + BMI + Polio + Diphtheria + Incidents_HIV + GDP_log + \
                    Pop_log + Thinness_ten_nineteen_years + Thinness_five_nine_years + Schooling + \
                    Infant_deaths + Child_deaths + Year_factor + Alcohol_consumption_cent + \
                    Hepatitis_B_cent + Measles_cent + BMI_cent + Polio_cent + Diphtheria_cent + \
                    Incidents_HIV_cent + GDP_log_cent + Pop_log_cent + Thinness_ten_nineteen_years_cent + \
                    Thinness_five_nine_years_cent + Schooling_cent + Infant_deaths_cent + Child_deaths_cent', \
                    groups=life_expectancy_nofrance['Country'], \
                    data=life_expectancy_nofrance)
life_expectancy_france_new = life_expectancy_nofrance.copy()  

ref_std = np.sqrt(lm_no_france.fit().cov_re.squeeze()) # std. deviation of random effects
res_std = np.sqrt(lm_no_france.fit().scale.squeeze()) # std. deviation of the idiosyncratic error
pred_no_france = lm_no_france.fit().predict() # fitted values
resid_no_france = lm_no_france.fit().resid; # residuals

# parametric bootstrap    
np.random.seed(123)
nb = 100  
boot_predict1 = pd.DataFrame(index=range(nb),columns=range(2000,2016))
for i in range(nb):
    ref_new = np.random.normal(0,ref_std,178).repeat(16) # generate new random effects
    resid_new = np.random.normal(0,res_std,2848) # generate new residuals
    life_expectancy_france_new['Life_expectancy'] = pred_no_france + ref_new + resid_new # new dataset
    lm_no_france_new = smf.mixedlm(formula='Life_expectancy ~ Economy_status + Region + Alcohol_consumption + \
                        Hepatitis_B + Measles + BMI + Polio + Diphtheria + Incidents_HIV + GDP_log + \
                        Pop_log + Thinness_ten_nineteen_years + Thinness_five_nine_years + Schooling + \
                        Infant_deaths + Child_deaths + Year_factor + Alcohol_consumption_cent + \
                        Hepatitis_B_cent + Measles_cent + BMI_cent + Polio_cent + Diphtheria_cent + \
                        Incidents_HIV_cent + GDP_log_cent + Pop_log_cent + Thinness_ten_nineteen_years_cent + \
                        Thinness_five_nine_years_cent + Schooling_cent + Infant_deaths_cent + Child_deaths_cent', \
                        groups=life_expectancy_france_new['Country'], \
                        data=life_expectancy_france_new)

    boot_predict1.iloc[i,:] = lm_no_france_new.fit().predict(life_expectancy_france) # new prediction for France
    
boot_predict1.quantile([0.025,0.975],0)

# residual cluster bootstrap
np.random.seed(123)
nb = 100   
boot_predict2 = pd.DataFrame(index=range(nb),columns=range(2000,2016))
for i in range(nb):
    ref_new = np.random.normal(0,ref_std,178).repeat(16) # generate new random effects
    resid_new = resid_no_france.iloc[np.random.choice([*range(0,len(life_expectancy_nofrance),16)],178, \
                                                      replace=True).repeat(16) + [*range(16)]*178] # resample the residuals by clusters
    life_expectancy_france_new['Life_expectancy'] = pred_no_france + ref_new + np.array(resid_new)
    lm_no_france_new = smf.mixedlm(formula='Life_expectancy ~ Economy_status + Region + Alcohol_consumption + \
                        Hepatitis_B + Measles + BMI + Polio + Diphtheria + Incidents_HIV + GDP_log + \
                        Pop_log + Thinness_ten_nineteen_years + Thinness_five_nine_years + Schooling + \
                        Infant_deaths + Child_deaths + Year_factor + Alcohol_consumption_cent + \
                        Hepatitis_B_cent + Measles_cent + BMI_cent + Polio_cent + Diphtheria_cent + \
                        Incidents_HIV_cent + GDP_log_cent + Pop_log_cent + Thinness_ten_nineteen_years_cent + \
                        Thinness_five_nine_years_cent + Schooling_cent + Infant_deaths_cent + Child_deaths_cent', \
                        groups=life_expectancy_france_new['Country'], \
                        data=life_expectancy_france_new)

    boot_predict2.iloc[i,:] = lm_no_france_new.fit().predict(life_expectancy_france)
    
boot_predict2.quantile([0.025,0.975],0)

# wild cluster bootstrap
np.random.seed(123)
nb = 100  
boot_predict3 = pd.DataFrame(index=range(nb),columns=range(2000,2016))
for i in range(nb):
    ref_new = np.random.normal(0,ref_std,178).repeat(16) # generate new random effects
    weights = (2*(np.random.uniform(0,1,178)).round(0)-1).repeat(16) # generate Rademacher weights
    resid_new = resid_no_france*weights # new residuals
    life_expectancy_france_new['Life_expectancy'] = pred_no_france + ref_new + np.array(resid_new)
    lm_no_france_new = smf.mixedlm(formula='Life_expectancy ~ Economy_status + Region + Alcohol_consumption + \
                        Hepatitis_B + Measles + BMI + Polio + Diphtheria + Incidents_HIV + GDP_log + \
                        Pop_log + Thinness_ten_nineteen_years + Thinness_five_nine_years + Schooling + \
                        Infant_deaths + Child_deaths + Year_factor + Alcohol_consumption_cent + \
                        Hepatitis_B_cent + Measles_cent + BMI_cent + Polio_cent + Diphtheria_cent + \
                        Incidents_HIV_cent + GDP_log_cent + Pop_log_cent + Thinness_ten_nineteen_years_cent + \
                        Thinness_five_nine_years_cent + Schooling_cent + Infant_deaths_cent + Child_deaths_cent', \
                        groups=life_expectancy_france_new['Country'], \
                        data=life_expectancy_france_new)

    boot_predict3.iloc[i,:] = lm_no_france_new.fit().predict(life_expectancy_france)
    
boot_predict3.quantile([0.025,0.975],0)

# pairs cluster bootstrap
np.random.seed(123)
nb = 100   
boot_predict4 = pd.DataFrame(index=range(nb),columns=range(2000,2016))

for i in range(nb):
    life_expectancy_france_new = life_expectancy_nofrance.iloc[np.random.choice([*range(0,len(life_expectancy_nofrance),16)],178, \
                                                                                replace=True).repeat(16) + [*range(16)]*178]
    lm_no_france_new = smf.mixedlm(formula='Life_expectancy ~ Economy_status + Region + Alcohol_consumption + \
                        Hepatitis_B + Measles + BMI + Polio + Diphtheria + Incidents_HIV + GDP_log + \
                        Pop_log + Thinness_ten_nineteen_years + Thinness_five_nine_years + Schooling + \
                        Infant_deaths + Child_deaths + Year_factor + Alcohol_consumption_cent + \
                        Hepatitis_B_cent + Measles_cent + BMI_cent + Polio_cent + Diphtheria_cent + \
                        Incidents_HIV_cent + GDP_log_cent + Pop_log_cent + Thinness_ten_nineteen_years_cent + \
                        Thinness_five_nine_years_cent + Schooling_cent + Infant_deaths_cent + Child_deaths_cent', \
                        groups=life_expectancy_france_new['Country'], \
                        data=life_expectancy_france_new)                   
    boot_predict4.iloc[i,:] = lm_no_france_new.fit().predict(life_expectancy_france)

boot_predict4.quantile([0.025,0.975],0)

# cross-validation

# creating the model matrix manually
dummies = pd.get_dummies(life_expectancy_sort[['Economy_status','Region','Year_factor']]) # create dummies for factor variables
life_expectancy_cv = pd.concat([life_expectancy_sort,dummies],axis = 1)

model_matrix = life_expectancy_cv.iloc[:,[3,*range(6,13),*range(15,18),20,21,22,*range(24,39),*range(42,6)]].astype(float)
model_matrix = pd.concat([model_matrix,pd.DataFrame(np.ones([len(model_matrix),1]))],axis = 1) # model matrix
model_response = life_expectancy_cv['Life_expectancy'] # vector of responses

np.random.seed(123)
rep = 100
folds = 10
rmse_cv =  pd.DataFrame(index=range(rep*folds),columns = ['mse'])
calib_cv = pd.DataFrame(index=range(rep*folds),columns = ['mse'])

from sklearn.model_selection import KFold
kf = KFold(n_splits=10) # create folds

k = 0
for i in range(rep):
    idx_cv = np.random.choice([*range(0,len(life_expectancy_cv),16)],179, replace=False) # reshuffle the observations
    
    for j, (train_index, test_index) in enumerate(kf.split(idx_cv)):
        
        train_set = idx_cv[train_index].repeat(16) + [*range(16)]*len(train_index) # extract train set indeces
        test_set = idx_cv[test_index].repeat(16) + [*range(16)]*len(test_index) # extract test set indeces

        # train set
        model_matrix_cv = model_matrix.iloc[train_set] 
        model_response_cv = model_response.iloc[train_set]
        model_country = life_expectancy_cv['Country'][train_set]
        
        # test set
        test_matrix_cv = model_matrix.iloc[test_set]
        test_response_cv = model_response.iloc[test_set]

        # check that some factor levels are not missing, i.e., whether there is a zero column in the model matrix
        # if it is, remove this column from the model matrix and the test observations
        zero_index = np.where(~model_matrix_cv.any(axis=0))[0]
        test_matrix_cv = test_matrix_cv.drop(model_matrix_cv.columns[zero_index], axis=1)
        model_matrix_cv = model_matrix_cv.drop(model_matrix_cv.columns[zero_index], axis=1)
        
        # fit the model
        lm_cv = sm.regression.mixed_linear_model.MixedLM(endog = model_response_cv, exog = model_matrix_cv, groups=model_country)

        # evaluate the model
        rmse_cv.iloc[k] = np.sqrt(((lm_cv.fit().predict(test_matrix_cv) - test_response_cv)**2).mean())
        calib_cv.iloc[k] = sm.OLS(endog = lm_cv.fit().predict(test_matrix_cv), exog = test_response_cv).fit().params.iloc[0]
        
        k = k +1

# cross-validation  RMSE      
rmse_cv.mean()
# cross-validation calibration
calib_cv.mean()

# cre model variance
np.sqrt(np.sqrt(lm_cr.fit().cov_re.squeeze())**2 + np.sqrt(lm_cr.fit().scale.squeeze())**2)

# cre model variance/cross-validation  RMSE
np.sqrt(np.sqrt(lm_cr.fit().cov_re.squeeze())**2 + np.sqrt(lm_cr.fit().scale.squeeze())**2)/rmse_cv.mean()