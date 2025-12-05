# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.model_selection import KFold


np.set_printoptions(legacy='1.25')


# load dataset
framingham = pd.read_csv('C:/Users/elini/Desktop/nine circles 2/framingham.csv')
framingham
framingham.dtypes

framingham['age'] = framingham['age'].astype(float)


# check duplicated
framingham.duplicated(keep = False).any(axis = None)

# check missing
framingham.isna().any(axis = None)
framingham.isna().sum(axis = 0)



# rename variables
framingham = framingham.rename(columns={'male': 'Sex', 'age': 'Age', 'currentSmoker': 'Smoker', \
                                        'prevalentStroke': 'Stroke', 'prevalentHyp': 'Hyp', \
                                        'diabetes': 'Diab', 'TenYearCHD': 'TCHD', 'sysBP': 'SysP', \
                                        'diaBP': 'DiaP', 'heartRate': 'Hrate', 'cigsPerDay': 'Cig', \
                                        'totChol': 'Chol', 'BPMeds': 'Meds', 'education': 'Edu', \
                                        'glucose': 'Gluc'})

framingham['Sex'].value_counts() # 0: female, 1:male
framingham['Edu'].value_counts()
framingham['Smoker'].value_counts()
framingham['Meds'].value_counts()
framingham['Stroke'].value_counts()
framingham['Hyp'].value_counts()
framingham['Diab'].value_counts()


fig, axs = plt.subplots(1,3)
axs[0].hist(framingham['Age'], bins=20)
axs[0].set_xlabel('Age')
axs[0].set_ylabel('Frequency')
axs[1].hist(framingham['SysP'], bins=20)  
axs[1].set_xlabel('SysP')    
axs[2].hist(framingham['DiaP'], bins=20)  
axs[2].set_xlabel('DiaP')    


fig, axs = plt.subplots(1,4)
axs[0].hist(framingham['Hrate'], bins=20)
axs[0].set_xlabel('Hrate')
axs[0].set_ylabel('Frequency')
axs[1].hist(framingham['Cig'], bins=20)  
axs[1].set_xlabel('Cig')    
axs[2].hist(framingham['Chol'], bins=20)  
axs[2].set_xlabel('Chol')
axs[3].hist(framingham['Gluc'], bins=20)  
axs[3].set_xlabel('Gluc')


# complete case analysis

framingham_complete = framingham.loc[~framingham.isna().any(axis = 1)].reset_index(drop=True)

from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF
ind = [*range(3),*range(4,15)]
VIF_dataframe = framingham_complete.iloc[:,ind].assign(const=1)

VIF_vals = [VIF(VIF_dataframe, i) 
        for i in range(0, VIF_dataframe.shape[1])]
VIF_vals = pd.DataFrame({'VIF':VIF_vals},index=VIF_dataframe.columns)
VIF_vals


lr_full = smf.logit(formula='TCHD ~ Sex + bs(Age, df=3) + C(Edu, Poly) + bs(Cig, df=3)  + Meds + Stroke + Hyp + Diab + bs(Chol, df=3) + \
                     bs(SysP, df=3) + bs(DiaP, df=3) + bs(BMI, df=3) + bs(Hrate, df=3) + bs(Gluc, df=3) + \
                     Age:(Cig + Stroke + Hyp + Diab + Chol + SysP + DiaP + BMI + Hrate + Gluc) + \
                     Sex:(Cig + Stroke + Hyp + Diab + Chol + SysP + DiaP + BMI + Hrate + Gluc)', data=framingham_complete)
lr_full_fit = lr_full.fit(disp=0)                              
print(lr_full_fit.summary())



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


# plot  effects plots
median_values = framingham_complete.median()
framingham_complete.min()
framingham_complete.max()


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
plt.legend()

# cig & sex
cig_seq = np.array([*range(0,80,10)])

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
plt.legend()


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
plt.legend()


# chol & sex
chol_seq = np.array([*range(120,600,60)])

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
plt.legend()


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
plt.legend()


# diap & sex
diap_seq = np.array([*range(50,140,10)])

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
plt.legend()


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
plt.legend()


# Hrate & sex
hrate_seq = np.array([*range(45,145,10)])

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
plt.legend()

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
plt.legend()

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
axs[1].set_ylabel('Predicted Probability of TCHD')
axs[1].set_title('Meds = 1')
axs[1].legend();


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
axs[1].set_ylabel('Predicted Probability of TCHD')
axs[1].set_title('Stroke = 1')
axs[1].legend()



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
axs[0].set_title('Meds = 0')
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
axs[1].set_ylabel('Predicted Probability of TCHD')
axs[1].set_title('Hyp = 1')
axs[1].legend();

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
axs[0].set_title('Meds = 0')
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


# test interactions

lr_no_inter = smf.logit(formula='TCHD ~ Sex + bs(Age, df=3) + C(Edu, Poly) + bs(Cig, df=3) + Meds + Stroke + Hyp + Diab + bs(Chol, df=3) + \
                     bs(SysP, df=3) + bs(DiaP, df=3) + bs(BMI, df=3) + bs(Hrate, df=3) + bs(Gluc, df=3)', \
                     data=framingham_complete)

1 - chi2.cdf(2*(lr_full_fit.llf-lr_no_inter.fit(disp=0).llf),(lr_full.df_model - lr_no_inter.df_model))

# test nonlinearities

lr_lin = smf.logit(formula='TCHD ~ Sex + Age + C(Edu, Poly) + Cig + Meds + Stroke + Hyp + Diab + Chol + \
                     SysP + DiaP + BMI + Hrate + Gluc + \
                     Age:(Cig + Stroke + Hyp + Diab + Chol + SysP + DiaP + BMI + Hrate + Gluc) + \
                     Sex:(Cig + Stroke + Hyp + Diab + Chol + SysP + DiaP + BMI + Hrate + Gluc)', data=framingham_complete)

1 - chi2.cdf(2*(lr_full_fit.llf-lr_lin.fit(disp=0).llf),(lr_full.df_model - lr_lin.df_model))


# which nonlinear terms are significant


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

# diagnostics

# raw residuals
plt.scatter(lr_full_fit.predict(), framingham_complete['TCHD'] - lr_full_fit.predict())
plt.title('Raw Residuals vs Predicted Probabilites')
plt.xlabel('Predicted Probabilites')
plt.ylabel('Raw Residuals')

sm.qqplot( framingham_complete['TCHD'] - lr_full_fit.predict(), line='s');


# deviance residuals
plt.scatter(lr_full_fit.predict(), lr_full_fit.resid_dev)
plt.title('Deviance Residuals vs Predicted Probabilites')
plt.xlabel('Predicted Probabilites')
plt.ylabel('Deviance Residuals')

sm.qqplot(lr_full_fit.resid_dev, line='s');

# Pearson residuals
pearson = pd.DataFrame(index = [*range(len(lr_full_fit.predict() ))], columns= ['Pearson'])
pearson['Pearson'] = ((framingham_complete['TCHD'] - \
                      lr_full_fit.predict())/np.sqrt(lr_full_fit.predict() * (1-lr_full_fit.predict())))
    
plt.scatter(lr_full_fit.predict(), pearson['Pearson'])
plt.title('Pearson Residuals vs Predicted Probabilites')
plt.xlabel('Pearson Probabilites')
plt.ylabel('Deviance Residuals')
    
    
pearson['Lin_pred'] = lr_full_fit.predict(which = 'linear')
sm.qqplot(pearson.groupby(pd.qcut(pearson['Lin_pred'], 20, labels=False)).mean()['Pearson'], line='s');


# predictive performance
from sklearn.metrics import (confusion_matrix, accuracy_score,\
                             roc_curve, roc_auc_score, log_loss, brier_score_loss)
    
from sklearn.calibration import calibration_curve   

# shrinkage

lr_triv = smf.logit(formula='TCHD ~ 1',data=framingham_complete)
lr_triv_fit = lr_triv.fit(disp=0) 

dev = 2*(lr_full_fit.llf - lr_triv_fit.llf)
(dev-len(lr_full_fit.params))/dev

# trivial model


lr_final = smf.logit(formula='TCHD ~ Sex + bs(Age, df=3) + C(Edu, Poly) + bs(Cig, df=3)  + Meds + Stroke + Hyp + Diab + bs(Chol, df=3) + \
                     bs(SysP, df=3) + bs(DiaP, df=3) + bs(BMI, df=3) + bs(Hrate, df=3) + bs(Gluc, df=3)',\
                     data=framingham_complete)
lr_final_fit = lr_final.fit(disp=0)     


dev = 2*(lr_final_fit.llf - lr_triv_fit.llf)
(dev-len(lr_final_fit.params))/dev


# predictions
y_pred_prob = lr_final_fit.predict()
y_obs = framingham_complete['TCHD'].to_numpy()

# confusion matrix
y_pred = (y_pred_prob > 0.5).astype(int)
confusion_matrix(y_obs,y_pred)


# accuracy
accuracy_score(y_obs,y_pred)

accuracy_scores = np.zeros(21)
accuracy_thresholds = np.zeros(21)

for k in range(21):
    accuracy_thresholds[k] = k/20
    accuracy_scores[k] = accuracy_score(y_obs,(y_pred_prob > accuracy_thresholds[k]).astype(int))
plt.scatter(accuracy_thresholds,accuracy_scores)   


# specificity & sensitivity
specificity = np.zeros(21)
sensitivity = np.zeros(21)

for k in range(21):
    conf_table = confusion_matrix(y_obs,(y_pred_prob > k/20).astype(int))
    specificity[k] = conf_table[0,0]/(conf_table[0,0] + conf_table[0,1])
    sensitivity[k] = conf_table[1,1]/(conf_table[1,0] + conf_table[1,1])
    
plt.scatter(accuracy_thresholds,specificity)  
plt.scatter(accuracy_thresholds,sensitivity)  

# ROC curve 
plt.scatter(1-specificity,sensitivity)

fpr, tpr, thresholds = roc_curve(y_obs,y_pred_prob)
plt.scatter(fpr,tpr)

# area under ROC/concordance index
roc_auc_score(y_obs,y_pred_prob)

# log score
(y_obs*np.log(y_pred_prob) + (1-y_obs)*np.log(1-y_pred_prob)).sum()
lr_final_fit.llf

-(y_obs*np.log(y_pred_prob) + (1-y_obs)*np.log(1-y_pred_prob)).mean()
log_loss(y_obs, y_pred_prob)

# Brier score
((y_obs-y_pred_prob)**2).mean()
brier_score_loss(y_obs,y_pred_prob)


# calibration

prob_true, prob_pred = calibration_curve(y_obs,y_pred_prob, n_bins=10, strategy = 'quantile')

plt.scatter(prob_true,prob_pred)
sm.OLS(endog = prob_pred, exog = pd.DataFrame(prob_true).assign(const=1)).fit(disp=0).params.iloc[0]


logitp = np.log(y_pred_prob/(1-y_pred_prob))

a = sm.Logit(endog= framingham_complete['TCHD'],exog = pd.DataFrame(logitp).assign(const=1)).fit(disp=0).params.iloc[0]
print(a.fit(disp=0).summary())


# DCA

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
plt.ylim(dca_curve[99]-0.01, dca_curve[0]+0.01)


# cross-validation
np.random.seed(123)

rep = 1
folds = 10
kf = KFold(n_splits=10) # create folds

metrics_cv =  pd.DataFrame(index=range(rep*folds),columns = ['AUC','Brier','log','cal'])
dca_curve_cv  = pd.DataFrame(index=range(rep*folds),columns = range(100))

k = 0
for i in range(rep):
    idx_cv = np.random.choice([*range(len(framingham_complete))],len(framingham_complete), replace=False)
    
    for j, (train_index, test_index) in enumerate(kf.split(idx_cv)):
        
        train_set = idx_cv[train_index]
        test_set = idx_cv[test_index]
        lr_cv_new = smf.logit(formula='TCHD ~ Sex + bs(Age, lower_bound=30, upper_bound = 75, df=3) + C(Edu, Poly) + \
                             bs(Cig, lower_bound= 0, upper_bound = 70, df=3) + Meds + Stroke + Hyp + Diab + bs(Chol, lower_bound= 100, upper_bound = 600, df=3) + \
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
        

metrics_cv.mean()

dca_curve_cv.mean()
plt.plot(dca_thresholds,dca_curve_cv.mean(),color = 'blue')
plt.plot(dca_thresholds,tnone_curve, color = 'red')
plt.plot(dca_thresholds,tall_curve, color = 'green')
plt.ylim(dca_curve[99]-0.01, dca_curve[0]+0.01)



## Mean/most_frequent imputation

from sklearn.impute import SimpleImputer

framingham.isna().any(axis = None)
framingham.isna().sum(axis = 0)


mean_imp_fit_cig = SimpleImputer(missing_values = np.nan, strategy='mean').fit(pd.DataFrame(framingham.loc[framingham['Smoker'] == 1]['Cig']))
mean_imp_fit = SimpleImputer(missing_values = np.nan, strategy='mean').fit(framingham[['Chol','BMI','Hrate','Gluc']])
mfreq_imp_fit = SimpleImputer(missing_values =np.nan, strategy="most_frequent").fit(framingham[['Edu','Meds']])

framingham_mean_imp = framingham.copy()
framingham_mean_imp[['Chol','BMI','Hrate','Gluc']] = \
    mean_imp_fit.transform(framingham_mean_imp[['Chol','BMI','Hrate','Gluc']])
framingham_mean_imp[['Cig']] = mean_imp_fit_cig.transform(framingham_mean_imp[['Cig']])
framingham_mean_imp[['Edu','Meds']] = mfreq_imp_fit.transform(framingham_mean_imp[['Edu','Meds']])
                   
                     
framingham_mean_imp.isna().any(axis = None)


lr_full_mean_imp = smf.logit(formula='TCHD ~ Sex + bs(Age, df=3) + C(Edu, Poly) + bs(Cig, df=3)  + Meds + Stroke + Hyp + Diab + bs(Chol, df=3) + \
                     bs(SysP, df=3) + bs(DiaP, df=3) + bs(BMI, df=3) + bs(Hrate, df=3) + bs(Gluc, df=3) + \
                     Age:(Cig + Stroke + Hyp + Diab + Chol + SysP + DiaP + BMI + Hrate + Gluc) + \
                     Sex:(Cig + Stroke + Hyp + Diab + Chol + SysP + DiaP + BMI + Hrate + Gluc)', data=framingham_mean_imp)                            

lr_full_mean_imp_fit = lr_full_mean_imp.fit(disp=0)
print(lr_full_mean_imp_fit.summary())


params = lr_full_mean_imp_fit.params.iloc[range(33,53)]
cov_params = lr_full_mean_imp_fit.cov_params().iloc[range(33,53)].iloc[:,range(33,53)]
wald_stat = np.transpose(params) @ np.linalg.inv(cov_params) @ params

params2 = lr_full_mean_imp_fit.params.iloc[[12,34,44]]
cov_params2 = lr_full_mean_imp_fit.cov_params().iloc[[12,34,44]].iloc[:,[12,34,44]]
wald_stat2 = np.transpose(params2) @ np.linalg.inv(cov_params2) @ params2

params3 = lr_full_mean_imp_fit.params.iloc[[*range(5,8),*range(33,43)]]
cov_params3 = lr_full_mean_imp_fit.cov_params().iloc[[*range(5,8),*range(33,43)]].iloc[:,[*range(5,8),*range(33,43)]]
wald_stat3 = np.transpose(params3) @ np.linalg.inv(cov_params3) @ params3



# pairs bootstrap wald test
np.random.seed(123)
nb = 500
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
    
    
(wald_stat <  wald_test).mean()
(wald_stat2 < wald_test2).mean()
(wald_stat3 < wald_test3).mean()

# cross-validation
np.random.seed(123)

rep = 1
folds = 10
kf = KFold(n_splits=10) # create folds

metrics_cv_mean_imp =  pd.DataFrame(index=range(rep*folds),columns = ['AUC','Brier','log','cal'])
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
        mfreq_imp_fit_new = SimpleImputer(missing_values =np.nan, strategy="most_frequent").fit(framingham_train[['Edu']])

        framingham_train_mean_imp = framingham_train.copy()
        framingham_train_mean_imp[['Cig']] = mean_imp_fit_cig_new.transform(framingham_train_mean_imp[['Cig']])
        framingham_train_mean_imp[['Chol','BMI','Hrate','Gluc']] = \
            mean_imp_fit_new.transform(framingham_train_mean_imp[['Chol','BMI','Hrate','Gluc']])
        framingham_train_mean_imp[['Edu','Meds']] = mfreq_imp_fit_new.transform(framingham_train_mean_imp[['Edu','Meds']])
            

        lr_cv_new = smf.logit(formula='TCHD ~ Sex + bs(Age, lower_bound=30, upper_bound = 75, df=3) + C(Edu, Poly) + \
                             bs(Cig, lower_bound= 0, upper_bound = 70, df=3) + Meds + Stroke + Hyp + Diab + bs(Chol, lower_bound= 100, upper_bound = 700, df=3) + \
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
        

metrics_cv_mean_imp.mean()

plt.plot(dca_thresholds,dca_curve_cv_mean_imp.mean(),color = 'blue')
plt.plot(dca_thresholds,tnone_curve, color = 'red')
plt.plot(dca_thresholds,tall_curve, color = 'green')
plt.ylim(dca_curve[99]-0.01, dca_curve[0]+0.01)


## knn 

from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

variables = ['Sex','Age','Edu','Cig','Smoker','Meds','Stroke','Hyp','Diab','Chol','SysP','DiaP','BMI','Hrate','Gluc']
predictors = framingham[variables]

scaler = StandardScaler().fit(predictors)
predictors_scaled = scaler.transform(predictors)
knn_imputer = KNNImputer(n_neighbors=10).fit(predictors_scaled)

framingham_knn_imp = framingham.copy()
framingham_knn_imp[variables] = scaler.inverse_transform(knn_imputer.transform(predictors_scaled))
framingham_knn_imp['Edu'] = round(framingham_knn_imp['Edu'])
framingham_knn_imp['Meds'] = round(framingham_knn_imp['Meds'])

lr_full_knn_imp = smf.logit(formula='TCHD ~ Sex + bs(Age, df=3) + C(Edu, Poly) + bs(Cig, df=3)  + Meds + Stroke + Hyp + Diab + bs(Chol, df=3) + \
                     bs(SysP, df=3) + bs(DiaP, df=3) + bs(BMI, df=3) + bs(Hrate, df=3) + bs(Gluc, df=3) + \
                     Age:(Cig + Stroke + Hyp + Diab + Chol + SysP + DiaP + BMI + Hrate + Gluc) + \
                     Sex:(Cig + Stroke + Hyp + Diab + Chol + SysP + DiaP + BMI + Hrate + Gluc)', data=framingham_knn_imp)                            

lr_full_knn_imp_fit = lr_full_knn_imp.fit(disp=0)
print(lr_full_knn_imp_fit.summary())



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
        

metrics_cv_knn_imp.mean()

plt.plot(dca_thresholds,dca_curve_cv_knn_imp.mean(),color = 'blue')
plt.plot(dca_thresholds,tnone_curve, color = 'red')
plt.plot(dca_thresholds,tall_curve, color = 'green')
plt.ylim(dca_curve[99]-0.01, dca_curve[0]+0.01)


## missforest 

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

warnings.filterwarnings('default')



metrics_cv_missforest_imp.mean()

plt.plot(dca_thresholds,dca_curve_cv_missforest_imp.mean(),color = 'blue')
plt.plot(dca_thresholds,tnone_curve, color = 'red')
plt.plot(dca_thresholds,tall_curve, color = 'green')
plt.ylim(dca_curve[99]-0.01, dca_curve[0]+0.01)


              