# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.model_selection import KFold

np.set_printoptions(legacy='1.25')


## Heart Disease Datasets https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data
## https://archive.ics.uci.edu/dataset/45/heart+disease


# load dataset
heart = pd.read_csv('C:/Users/elini/Desktop/nine circles 2/heart_disease_uci.csv')
heart
heart.dtypes

heart.loc[heart['chol'] == 0,'chol'] = np.nan
heart.loc[heart['trestbps'] == 0,'trestbps'] = np.nan


heart.iloc[:,range(1,16)].duplicated(keep = False).any(axis = None)
heart.loc[heart.iloc[:,range(1,16)].duplicated(keep = False) == True]
heart_red = heart.drop(axis = 0, index = [405,907]).reset_index()


heart_red.isna().sum(axis = 0)


# numerical predictors

numeric_variables_heart = ['age','trestbps','chol','thalch','oldpeak']
categorical_variables_heart = ['sex','dataset','cp','fbs','restecg','exang','slope','thal']

numeric_basic_stats = np.zeros([len(numeric_variables_heart),6])

for i in range(len(numeric_variables_heart)):
    numeric_basic_stats[i,0] = np.mean(heart_red[numeric_variables_heart[i]])
    numeric_basic_stats[i,1] = np.nanmedian(heart_red[numeric_variables_heart[i]])
    numeric_basic_stats[i,2] = np.nanmin(heart_red[numeric_variables_heart[i]])
    numeric_basic_stats[i,3] = np.nanquantile(heart_red[numeric_variables_heart[i]], 0.25)
    numeric_basic_stats[i,4] = np.nanquantile(heart_red[numeric_variables_heart[i]], 0.75)
    numeric_basic_stats[i,5] = np.nanmax(heart_red[numeric_variables_heart[i]])

      
pd.DataFrame(data=numeric_basic_stats, index=numeric_variables_heart, columns=['mean','median','min','1st qrt.','3rd qrt.','max'])

f,a = plt.subplots(1,3)
a = a.ravel()
for idx,ax in enumerate(a):
    ax.hist(heart_red[numeric_variables_heart[idx]], bins=20)
    ax.set_title('Histogram')
    ax.set_xlabel(numeric_variables_heart[idx])
    ax.set_ylabel('Frequency')
plt.tight_layout()


f,a = plt.subplots(1,2)
a = a.ravel()
for idx,ax in enumerate(a):
    ax.hist(heart_red[numeric_variables_heart[idx+3]], bins=20)
    ax.set_title('Histogram')
    ax.set_xlabel(numeric_variables_heart[idx+3])
    ax.set_ylabel('Frequency')
plt.tight_layout()


# categorical predictors

heart_red['sex'].value_counts()
heart_red['dataset'].value_counts()
heart_red['cp'].value_counts()
heart_red['fbs'].value_counts()
heart_red['restecg'].value_counts()
heart_red['exang'].value_counts()
heart_red['slope'].value_counts()
heart_red['thal'].value_counts()


# imputation

heart_imp = heart_red[['age','sex','dataset','cp','trestbps','chol','fbs','restecg','thalch','exang','oldpeak','slope', 'thal']].copy()

heart_imp['sex'] = heart_imp['sex'].map({'Male': 1, 'Female': 0})
heart_imp['dataset'] = heart_imp['dataset'].map({'Cleveland': 0, 'Hungary': 1, 'Switzerland': 2, 'VA Long Beach': 3})
heart_imp['cp'] = heart_imp['cp'].map({'asymptomatic': 0, 'typical angina': 1, 'atypical angina': 2, 'non-anginal': 3})
heart_imp['fbs'] = heart_imp['fbs'].map({True: 1, False: 0})
heart_imp['restecg'] = heart_imp['restecg'].map({'normal': 0, 'lv hypertrophy': 1, 'st-t abnormality': 2})
heart_imp['exang'] = heart_imp['exang'].map({True: 1, False: 0})
heart_imp['slope'] = heart_imp['slope'].map({'flat': 0, 'downsloping': 1, 'upsloping': 2})
heart_imp['thal'] = heart_imp['thal'].map({'normal': 0, 'reversable defect': 1, 'fixed defect': 2})


from missforest import MissForest
np.random.seed(123)

MissForest_imputation = MissForest(categorical=categorical_variables_heart)
MissForest_imputation._verbose = 0

MissForest_imputation.fit(x = heart_imp)
heart_imp_MissForest = heart_imp.copy()

heart_imp_MissForest =  MissForest_imputation.transform(x = heart_imp)


# dummies

heart_final = heart_imp_MissForest.copy()
heart_final['heart_disease'] = (heart['num'] > 0).astype(int) 
heart_final['dataset'] = heart_final['dataset'].map({0: 'Clv', 1: 'Hun', 2: 'Swit', 3: 'VA'})
heart_final['cp'] = heart_final['cp'].map({0: 'asymp', 1: 'typ', 2: 'atyp', 3: 'nonang'})
heart_final['restecg'] = heart_final['restecg'].map({0: 'normal', 1: 'hypertrophy', 2: 'stt'})
heart_final['slope'] = heart_final['slope'].map({0: 'flat', 1: 'down', 2: 'up'})
heart_final['thal'] = heart_final['thal'].map({0: 'normal', 1: 'rev', 2: 'fixed'})

heart_final = pd.concat([heart_final,pd.get_dummies(heart_final[['cp','restecg','slope','thal']], dtype=int)], axis=1)
heart_final = heart_final[['age','sex','cp_typ','cp_atyp','cp_nonang','restecg_hypertrophy','restecg_stt',\
                           'thalch','exang','trestbps','oldpeak','fbs','chol','slope_down','slope_up','thal_rev','thal_fixed','heart_disease']]


# logistic regression

log_reg_heart = smf.logit(formula='heart_disease ~ age + sex + cp_typ + cp_atyp + cp_nonang + restecg_hypertrophy + \
                          restecg_stt + thalch + exang + trestbps + oldpeak + fbs + chol + slope_down + slope_up + \
                          thal_rev + thal_fixed', data=heart_final)
log_reg_heart_fit = log_reg_heart.fit(disp=0)                              
print(log_reg_heart_fit.summary())


pred_prob_heart = log_reg_heart_fit.predict()
obs_heart = heart_final['heart_disease'].to_numpy()



from sklearn.metrics import (accuracy_score, balanced_accuracy_score, roc_curve, roc_auc_score, log_loss, brier_score_loss, f1_score, matthews_corrcoef, cohen_kappa_score)
from sklearn.calibration import calibration_curve  

pred_prob_heart = log_reg_heart_fit.predict()
obs_heart = heart_final['heart_disease'].to_numpy()

# scores
accuracy_scores_lr = np.zeros(41)
balanced_accuracy_scores_lr = np.zeros(41)
f1_scores_lr = np.zeros(41)
f1_scores_neg_lr = np.zeros(41)
matthews_corrcoefs_lr = np.zeros(41)
cohen_kappa_scores_lr = np.zeros(41)
accuracy_thresholds = np.zeros(41)


for k in range(41):
    accuracy_thresholds[k] = k/40
    accuracy_scores_lr[k] = accuracy_score(obs_heart,(pred_prob_heart > accuracy_thresholds[k]).astype(int))
    balanced_accuracy_scores_lr[k] = balanced_accuracy_score(obs_heart,(pred_prob_heart > accuracy_thresholds[k]).astype(int))
    f1_scores_lr[k] = f1_score(obs_heart,(pred_prob_heart > accuracy_thresholds[k]).astype(int))
    f1_scores_neg_lr[k] = f1_score(obs_heart,(pred_prob_heart > accuracy_thresholds[k]).astype(int),pos_label=0)
    matthews_corrcoefs_lr[k] = matthews_corrcoef(obs_heart,(pred_prob_heart > accuracy_thresholds[k]).astype(int))
    cohen_kappa_scores_lr[k] = cohen_kappa_score(obs_heart,(pred_prob_heart > accuracy_thresholds[k]).astype(int))
    
    

plt.plot(accuracy_thresholds,accuracy_scores_lr)
plt.plot(accuracy_thresholds,balanced_accuracy_scores_lr)
plt.plot(accuracy_thresholds,f1_scores_lr)
plt.plot(accuracy_thresholds,f1_scores_neg_lr)
plt.plot(accuracy_thresholds,matthews_corrcoefs_lr)
plt.plot(accuracy_thresholds,cohen_kappa_scores_lr)

plt.legend(['Accuracy','Balanced Accuracy','F1-score','F1-score (for negatives)','Matthews correlation',"Cohen's kappa"], loc="lower right")
plt.xlabel('Probability Threshold')
plt.ylabel('Metrics');


fpr, tpr, thresholds = roc_curve(obs_heart,pred_prob_heart)
plt.plot(fpr,tpr)
plt.xlabel('1 - Specificity (false positive rate)')
plt.ylabel('Sensitivity (true positive rate)')


roc_auc_score(obs_heart,pred_prob_heart)
log_loss(obs_heart, pred_prob_heart)
brier_score_loss(obs_heart,pred_prob_heart)

# for a comparison
brier_score_loss(obs_heart,(heart_final['heart_disease'].sum()/len(heart_final)).repeat(len(heart_final)))


prob_true, prob_pred = calibration_curve(obs_heart,pred_prob_heart, n_bins=10, strategy = 'quantile')
sm.graphics.abline_plot(intercept = 0,slope = 1,color = 'red')
plt.scatter(prob_true,prob_pred)
plt.xlabel('Observed Probability')
plt.ylabel('Predicted Probability')
    
# LDA

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


lda_heart = LinearDiscriminantAnalysis(store_covariance=True)
lda_heart_fit = lda_heart.fit(X = heart_final.iloc[:,range(0,17)], y = heart_final['heart_disease']) 


pred_prob_heart_lda = lda_heart_fit.predict_proba(heart_final.iloc[:,range(0,17)]) # predict probabilities
pred_class_heart_lda_class = lda_heart_fit.predict(heart_final.iloc[:,range(0,17)]) # predict class
(pred_prob_heart_lda[:,1] > 0.5).astype(int) - pred_class_heart_lda_class
    

lda_heart_fit.classes_  # classes
lda_heart_fit.means_    # means 
lda_heart_fit.covariance_ # covariance
lda_heart_fit.priors_  # priors


# scores
accuracy_scores_lda = np.zeros(41)
balanced_accuracy_scores_lda = np.zeros(41)
f1_scores_lda = np.zeros(41)
f1_scores_neg_lda = np.zeros(41)
matthews_corrcoefs_lda = np.zeros(41)
cohen_kappa_scores_lda = np.zeros(41)

for k in range(41):
    accuracy_thresholds[k] = k/40
    accuracy_scores_lda[k] = accuracy_score(obs_heart,(pred_prob_heart_lda[:,1] > accuracy_thresholds[k]).astype(int))
    balanced_accuracy_scores_lda[k] = balanced_accuracy_score(obs_heart,(pred_prob_heart_lda[:,1] > accuracy_thresholds[k]).astype(int))
    f1_scores_lda[k] = f1_score(obs_heart,(pred_prob_heart_lda[:,1] > accuracy_thresholds[k]).astype(int))
    f1_scores_neg_lda[k] = f1_score(obs_heart,(pred_prob_heart_lda[:,1] > accuracy_thresholds[k]).astype(int),pos_label=0)
    matthews_corrcoefs_lda[k] = matthews_corrcoef(obs_heart,(pred_prob_heart_lda[:,1] > accuracy_thresholds[k]).astype(int))
    cohen_kappa_scores_lda[k] = cohen_kappa_score(obs_heart,(pred_prob_heart_lda[:,1] > accuracy_thresholds[k]).astype(int))


plt.plot(accuracy_thresholds,accuracy_scores_lda)
plt.plot(accuracy_thresholds,balanced_accuracy_scores_lda)
plt.plot(accuracy_thresholds,f1_scores_lda)
plt.plot(accuracy_thresholds,f1_scores_neg_lda)
plt.plot(accuracy_thresholds,matthews_corrcoefs_lda)
plt.plot(accuracy_thresholds,cohen_kappa_scores_lda)

plt.legend(['Accuracy','Balanced Accuracy','F1-score','F1-score (for negatives)','Matthews correlation',"Cohen's kappa"], loc="lower right")
plt.xlabel('Probability Threshold')
plt.ylabel('Metrics');

fpr, tpr, thresholds = roc_curve(obs_heart,pred_prob_heart_lda[:,1])
plt.plot(fpr,tpr)
plt.xlabel('1 - Specificity (false positive rate)')
plt.ylabel('Sensitivity (true positive rate)');


roc_auc_score(obs_heart,pred_prob_heart_lda[:,1])
log_loss(obs_heart, pred_prob_heart_lda[:,1])
brier_score_loss(obs_heart,pred_prob_heart_lda[:,1])


prob_true_lda, prob_pred_lda = calibration_curve(obs_heart,pred_prob_heart_lda[:,1], n_bins=10, strategy = 'quantile')
sm.graphics.abline_plot(intercept = 0,slope = 1,color = 'red')
plt.scatter(prob_true_lda,prob_pred_lda)
plt.xlabel('Observed Probability')
plt.ylabel('Predicted Probability')


# SVM


from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

svm_pipeline_heart1 = make_pipeline(StandardScaler(),  svm.SVC(C=1.0, kernel = 'linear'))
svm_pipeline_heart_fit1 = svm_pipeline_heart1.fit(X = heart_final.iloc[:,range(0,17)], y = heart_final['heart_disease'])
svm_predict_heart1 = svm_pipeline_heart_fit1.predict(heart_final.iloc[:,range(0,17)])

from sklearn.metrics import confusion_matrix
confusion_matrix(obs_heart,svm_predict_heart1) 


svm_pipeline_heart2 = make_pipeline(StandardScaler(),  svm.SVC(C=100.0, kernel = 'linear'))
svm_pipeline_heart_fit2 = svm_pipeline_heart2.fit(X = heart_final.iloc[:,range(0,17)], y = heart_final['heart_disease'])
svm_predict_heart2 = svm_pipeline_heart_fit2.predict(heart_final.iloc[:,range(0,17)])

confusion_matrix(obs_heart,svm_predict_heart2) 

# estimate optimal C

from sklearn.model_selection import GridSearchCV

parameters = {'kernel':['linear'], 'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000]}


svm_pipeline_heart3 = make_pipeline(StandardScaler(),  GridSearchCV(estimator=svm.SVC(),param_grid=parameters, scoring = 'balanced_accuracy'))

svm_pipeline_heart_fit3 = svm_pipeline_heart3.fit(X = heart_final.iloc[:,range(0,17)], y = heart_final['heart_disease'])

svm_predict_heart3 = svm_pipeline_heart_fit3.predict(heart_final.iloc[:,range(0,17)])


svm_pipeline_heart_fit3._final_estimator
svm_pipeline_heart_fit3._final_estimator.best_estimator_
svm_pipeline_heart_fit3._final_estimator.cv_results_


# platt scaling
svm_pipeline_heart_best = make_pipeline(StandardScaler(),  svm.SVC(C=0.01, kernel = 'linear', probability = True))
svm_pipeline_heart_fit_best = svm_pipeline_heart_best.fit(X = heart_final.iloc[:,range(0,17)], y = heart_final['heart_disease'])
pred_prob_heart_svm = svm_pipeline_heart_fit_best.predict_proba(heart_final.iloc[:,range(0,17)])


accuracy_scores_svm = np.zeros(41)
balanced_accuracy_scores_svm = np.zeros(41)
f1_scores_svm = np.zeros(41)
f1_scores_neg_svm = np.zeros(41)
matthews_corrcoefs_svm = np.zeros(41)
cohen_kappa_scores_svm = np.zeros(41)

for k in range(41):
    accuracy_thresholds[k] = k/40
    accuracy_scores_svm[k] = accuracy_score(obs_heart,(pred_prob_heart_svm[:,1] > accuracy_thresholds[k]).astype(int))
    balanced_accuracy_scores_svm[k] = balanced_accuracy_score(obs_heart,(pred_prob_heart_svm[:,1] > accuracy_thresholds[k]).astype(int))
    f1_scores_svm[k] = f1_score(obs_heart,(pred_prob_heart_svm[:,1] > accuracy_thresholds[k]).astype(int))
    f1_scores_neg_svm[k] = f1_score(obs_heart,(pred_prob_heart_svm[:,1] > accuracy_thresholds[k]).astype(int),pos_label=0)
    matthews_corrcoefs_svm[k] = matthews_corrcoef(obs_heart,(pred_prob_heart_svm[:,1] > accuracy_thresholds[k]).astype(int))
    cohen_kappa_scores_svm[k] = cohen_kappa_score(obs_heart,(pred_prob_heart_svm[:,1] > accuracy_thresholds[k]).astype(int))
    
    
plt.plot(accuracy_thresholds,accuracy_scores_svm)
plt.plot(accuracy_thresholds,balanced_accuracy_scores_svm)
plt.plot(accuracy_thresholds,f1_scores_svm)
plt.plot(accuracy_thresholds,f1_scores_neg_svm)
plt.plot(accuracy_thresholds,matthews_corrcoefs_svm)
plt.plot(accuracy_thresholds,cohen_kappa_scores_svm)

plt.legend(['Accuracy','Balanced Accuracy','F1-score','F1-score (for negatives)','Matthews correlation',"Cohen's kappa"], loc="lower right")
plt.xlabel('Probability Threshold')
plt.ylabel('Metrics')


fpr, tpr, thresholds = roc_curve(obs_heart,pred_prob_heart_svm[:,1])
plt.plot(fpr,tpr)
plt.xlabel('1 - Specificity (false positive rate)')
plt.ylabel('Sensitivity (true positive rate)')

roc_auc_score(obs_heart,pred_prob_heart_svm[:,1])
log_loss(obs_heart, pred_prob_heart_svm[:,1])
brier_score_loss(obs_heart,pred_prob_heart_svm[:,1])


prob_true_svm, prob_pred_svm = calibration_curve(obs_heart,pred_prob_heart_svm[:,1], n_bins=10, strategy = 'quantile')
sm.graphics.abline_plot(intercept = 0,slope = 1,color = 'red')
plt.scatter(prob_true_svm,prob_pred_svm)
plt.xlabel('Observed Probability')
plt.ylabel('Predicted Probability')


# kernel trick

# rbf
parameters_rbf = {'kernel':['rbf'], 'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000], 'gamma':[0.001, 0.01, 0.1, 1, 10, 100, 1000]}
svm_pipeline_heart_rbf = make_pipeline(StandardScaler(),  GridSearchCV(estimator=svm.SVC(probability=True),param_grid=parameters_rbf, scoring = 'balanced_accuracy'))
svm_pipeline_heart_rbf_fit = svm_pipeline_heart_rbf.fit(X = heart_final.iloc[:,range(0,17)], y = heart_final['heart_disease'])
svm_pipeline_heart_rbf_fit._final_estimator.best_estimator_

pred_prob_heart_svm_rbf = svm_pipeline_heart_rbf_fit.predict_proba(heart_final.iloc[:,range(0,17)])

accuracy_scores_svm_rbf = np.zeros(41)
balanced_accuracy_scores_svm_rbf = np.zeros(41)
f1_scores_svm_rbf = np.zeros(41)
f1_scores_neg_svm_rbf = np.zeros(41)
matthews_corrcoefs_svm_rbf = np.zeros(41)
cohen_kappa_scores_svm_rbf = np.zeros(41)

for k in range(41):
    accuracy_thresholds[k] = k/40
    accuracy_scores_svm_rbf[k] = accuracy_score(obs_heart,(pred_prob_heart_svm_rbf[:,1] > accuracy_thresholds[k]).astype(int))
    balanced_accuracy_scores_svm_rbf[k] = balanced_accuracy_score(obs_heart,(pred_prob_heart_svm_rbf[:,1] > accuracy_thresholds[k]).astype(int))
    f1_scores_svm_rbf[k] = f1_score(obs_heart,(pred_prob_heart_svm_rbf[:,1] > accuracy_thresholds[k]).astype(int))
    f1_scores_neg_svm_rbf[k] = f1_score(obs_heart,(pred_prob_heart_svm_rbf[:,1] > accuracy_thresholds[k]).astype(int),pos_label=0)
    matthews_corrcoefs_svm_rbf[k] = matthews_corrcoef(obs_heart,(pred_prob_heart_svm_rbf[:,1] > accuracy_thresholds[k]).astype(int))
    cohen_kappa_scores_svm_rbf[k] = cohen_kappa_score(obs_heart,(pred_prob_heart_svm_rbf[:,1] > accuracy_thresholds[k]).astype(int))
    
    
plt.plot(accuracy_thresholds,accuracy_scores_svm_rbf)
plt.plot(accuracy_thresholds,balanced_accuracy_scores_svm_rbf)
plt.plot(accuracy_thresholds,f1_scores_svm_rbf)
plt.plot(accuracy_thresholds,f1_scores_neg_svm_rbf)
plt.plot(accuracy_thresholds,matthews_corrcoefs_svm_rbf)
plt.plot(accuracy_thresholds,cohen_kappa_scores_svm_rbf)

plt.legend(['Accuracy','Balanced Accuracy','F1-score','F1-score (for negatives)','Matthews correlation',"Cohen's kappa"], loc="lower right")
plt.xlabel('Probability Threshold')
plt.ylabel('Metrics')


fpr, tpr, thresholds = roc_curve(obs_heart,pred_prob_heart_svm_rbf[:,1])
plt.plot(fpr,tpr)
plt.xlabel('1 - Specificity (false positive rate)')
plt.ylabel('Sensitivity (true positive rate)')


roc_auc_score(obs_heart,pred_prob_heart_svm_rbf[:,1])
log_loss(obs_heart, pred_prob_heart_svm_rbf[:,1])
brier_score_loss(obs_heart,pred_prob_heart_svm_rbf[:,1])

prob_true_svm_rbf, prob_pred_svm_rbf = calibration_curve(obs_heart,pred_prob_heart_svm_rbf[:,1], n_bins=10, strategy = 'quantile')
sm.graphics.abline_plot(intercept = 0,slope = 1,color = 'red')
plt.scatter(prob_true_svm_rbf,prob_pred_svm_rbf)
plt.xlabel('Observed Probability')
plt.ylabel('Predicted Probability')

# poly
parameters_poly = {'kernel':['poly'], 'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000], 'gamma':[0.001, 0.01, 0.1, 1, 10, 100, 1000],'coef0':[0,1,10], 'degree':[3]}
svm_pipeline_heart_poly = make_pipeline(StandardScaler(),  GridSearchCV(estimator=svm.SVC(probability=True),param_grid=parameters_poly, scoring = 'balanced_accuracy'))
svm_pipeline_heart_poly_fit = svm_pipeline_heart_poly.fit(X = heart_final.iloc[:,range(0,17)], y = heart_final['heart_disease'])
svm_pipeline_heart_poly_fit._final_estimator.best_estimator_


pred_prob_heart_svm_poly = svm_pipeline_heart_poly_fit.predict_proba(heart_final.iloc[:,range(0,17)])

accuracy_scores_svm_poly = np.zeros(41)
balanced_accuracy_scores_svm_poly = np.zeros(41)
f1_scores_svm_poly = np.zeros(41)
f1_scores_neg_svm_poly = np.zeros(41)
matthews_corrcoefs_svm_poly = np.zeros(41)
cohen_kappa_scores_svm_poly = np.zeros(41)

for k in range(41):
    accuracy_thresholds[k] = k/40
    accuracy_scores_svm_poly[k] = accuracy_score(obs_heart,(pred_prob_heart_svm_poly[:,1] > accuracy_thresholds[k]).astype(int))
    balanced_accuracy_scores_svm_poly[k] = balanced_accuracy_score(obs_heart,(pred_prob_heart_svm_poly[:,1] > accuracy_thresholds[k]).astype(int))
    f1_scores_svm_poly[k] = f1_score(obs_heart,(pred_prob_heart_svm_poly[:,1] > accuracy_thresholds[k]).astype(int))
    f1_scores_neg_svm_poly[k] = f1_score(obs_heart,(pred_prob_heart_svm_poly[:,1] > accuracy_thresholds[k]).astype(int),pos_label=0)
    matthews_corrcoefs_svm_poly[k] = matthews_corrcoef(obs_heart,(pred_prob_heart_svm_poly[:,1] > accuracy_thresholds[k]).astype(int))
    cohen_kappa_scores_svm_poly[k] = cohen_kappa_score(obs_heart,(pred_prob_heart_svm_poly[:,1] > accuracy_thresholds[k]).astype(int))

plt.plot(accuracy_thresholds,accuracy_scores_svm_poly)
plt.plot(accuracy_thresholds,balanced_accuracy_scores_svm_poly)
plt.plot(accuracy_thresholds,f1_scores_svm_poly)
plt.plot(accuracy_thresholds,f1_scores_neg_svm_poly)
plt.plot(accuracy_thresholds,matthews_corrcoefs_svm_poly)
plt.plot(accuracy_thresholds,cohen_kappa_scores_svm_poly)

plt.legend(['Accuracy','Balanced Accuracy','F1-score','F1-score (for negatives)','Matthews correlation',"Cohen's kappa"], loc="lower right")
plt.xlabel('Probability Threshold')
plt.ylabel('Metrics')


fpr, tpr, thresholds = roc_curve(obs_heart,pred_prob_heart_svm_poly[:,1])
plt.plot(fpr,tpr)
plt.xlabel('1 - Specificity (false positive rate)')
plt.ylabel('Sensitivity (true positive rate)')


roc_auc_score(obs_heart,pred_prob_heart_svm_poly[:,1])
log_loss(obs_heart, pred_prob_heart_svm_poly[:,1])
brier_score_loss(obs_heart,pred_prob_heart_svm_poly[:,1])

prob_true_svm_poly, prob_pred_svm_poly = calibration_curve(obs_heart,pred_prob_heart_svm_poly[:,1], n_bins=10, strategy = 'quantile')
sm.graphics.abline_plot(intercept = 0,slope = 1,color = 'red')
plt.scatter(prob_true_svm_poly,prob_pred_svm_poly)
plt.xlabel('Observed Probability')
plt.ylabel('Predicted Probability')


# sigmoid


parameters_sig = {'kernel':['sigmoid'], 'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000], 'gamma':[0.001, 0.01, 0.1, 1, 10, 100, 1000],'coef0':[-10, -1, 0.1, 0, 0.1, 1, 10]}

svm_pipeline_heart_sig = make_pipeline(StandardScaler(),  GridSearchCV(estimator=svm.SVC(probability=True),param_grid=parameters_sig, scoring = 'balanced_accuracy'))
svm_pipeline_heart_sig_fit = svm_pipeline_heart_sig.fit(X = heart_final.iloc[:,range(0,17)], y = heart_final['heart_disease'])
svm_pipeline_heart_sig_fit._final_estimator.best_estimator_


pred_prob_heart_svm_sig = svm_pipeline_heart_sig_fit.predict_proba(heart_final.iloc[:,range(0,17)])

accuracy_scores_svm_sig = np.zeros(41)
balanced_accuracy_scores_svm_sig = np.zeros(41)
f1_scores_svm_sig = np.zeros(41)
f1_scores_neg_svm_sig = np.zeros(41)
matthews_corrcoefs_svm_sig = np.zeros(41)
cohen_kappa_scores_svm_sig = np.zeros(41)

for k in range(41):
    accuracy_thresholds[k] = k/40
    accuracy_scores_svm_sig[k] = accuracy_score(obs_heart,(pred_prob_heart_svm_sig[:,1] > accuracy_thresholds[k]).astype(int))
    balanced_accuracy_scores_svm_sig[k] = balanced_accuracy_score(obs_heart,(pred_prob_heart_svm_sig[:,1] > accuracy_thresholds[k]).astype(int))
    f1_scores_svm_sig[k] = f1_score(obs_heart,(pred_prob_heart_svm_sig[:,1] > accuracy_thresholds[k]).astype(int))
    f1_scores_neg_svm_sig[k] = f1_score(obs_heart,(pred_prob_heart_svm_sig[:,1] > accuracy_thresholds[k]).astype(int),pos_label=0)
    matthews_corrcoefs_svm_sig[k] = matthews_corrcoef(obs_heart,(pred_prob_heart_svm_sig[:,1] > accuracy_thresholds[k]).astype(int))
    cohen_kappa_scores_svm_sig[k] = cohen_kappa_score(obs_heart,(pred_prob_heart_svm_sig[:,1] > accuracy_thresholds[k]).astype(int))
    
    
plt.plot(accuracy_thresholds,accuracy_scores_svm_sig)
plt.plot(accuracy_thresholds,balanced_accuracy_scores_svm_sig)
plt.plot(accuracy_thresholds,f1_scores_svm_sig)
plt.plot(accuracy_thresholds,f1_scores_neg_svm_sig)
plt.plot(accuracy_thresholds,matthews_corrcoefs_svm_sig)
plt.plot(accuracy_thresholds,cohen_kappa_scores_svm_sig)

plt.legend(['Accuracy','Balanced Accuracy','F1-score','F1-score (for negatives)','Matthews correlation',"Cohen's kappa"], loc="lower right")
plt.xlabel('Probability Threshold')
plt.ylabel('Metrics')

fpr, tpr, thresholds = roc_curve(obs_heart,pred_prob_heart_svm_sig[:,1])
plt.plot(fpr,tpr)
plt.xlabel('1 - Specificity (false positive rate)')
plt.ylabel('Sensitivity (true positive rate)')

roc_auc_score(obs_heart,pred_prob_heart_svm_sig[:,1])
log_loss(obs_heart, pred_prob_heart_svm_sig[:,1])
brier_score_loss(obs_heart,pred_prob_heart_svm_sig[:,1])

prob_true_svm_sig, prob_pred_svm_sig = calibration_curve(obs_heart,pred_prob_heart_svm_sig[:,1], n_bins=10, strategy = 'quantile')
sm.graphics.abline_plot(intercept = 0,slope = 1,color = 'red')
plt.scatter(prob_true_svm_sig,prob_pred_svm_sig)
plt.xlabel('Observed Probability')
plt.ylabel('Predicted Probability')



# QDA

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

qda_heart = QuadraticDiscriminantAnalysis(store_covariance=True) # store_covariance=True to explicitly compute the covariance matrices
qda_heart_fit = qda_heart.fit(X = heart_final.iloc[:,range(0,17)], y = heart_final['heart_disease']) 

pred_prob_heart_qda = qda_heart_fit.predict_proba(heart_final.iloc[:,range(0,17)])

len(qda_heart_fit.covariance_)


accuracy_scores_qda = np.zeros(41)
balanced_accuracy_scores_qda = np.zeros(41)
f1_scores_qda = np.zeros(41)
f1_scores_neg_qda = np.zeros(41)
matthews_corrcoefs_qda = np.zeros(41)
cohen_kappa_scores_qda = np.zeros(41)

for k in range(41):
    accuracy_thresholds[k] = k/40
    accuracy_scores_qda[k] = accuracy_score(obs_heart,(pred_prob_heart_qda[:,1] > accuracy_thresholds[k]).astype(int))
    balanced_accuracy_scores_qda[k] = balanced_accuracy_score(obs_heart,(pred_prob_heart_qda[:,1] > accuracy_thresholds[k]).astype(int))
    f1_scores_qda[k] = f1_score(obs_heart,(pred_prob_heart_qda[:,1] > accuracy_thresholds[k]).astype(int))
    f1_scores_neg_qda[k] = f1_score(obs_heart,(pred_prob_heart_qda[:,1] > accuracy_thresholds[k]).astype(int),pos_label=0)
    matthews_corrcoefs_qda[k] = matthews_corrcoef(obs_heart,(pred_prob_heart_qda[:,1] > accuracy_thresholds[k]).astype(int))
    cohen_kappa_scores_qda[k] = cohen_kappa_score(obs_heart,(pred_prob_heart_qda[:,1] > accuracy_thresholds[k]).astype(int))
    
plt.plot(accuracy_thresholds,accuracy_scores_qda)
plt.plot(accuracy_thresholds,balanced_accuracy_scores_qda)
plt.plot(accuracy_thresholds,f1_scores_qda)
plt.plot(accuracy_thresholds,f1_scores_neg_qda)
plt.plot(accuracy_thresholds,matthews_corrcoefs_qda)
plt.plot(accuracy_thresholds,cohen_kappa_scores_qda)

plt.legend(['Accuracy','Balanced Accuracy','F1-score','F1-score (for negatives)','Matthews correlation',"Cohen's kappa"], loc="lower right")
plt.xlabel('Probability Threshold')
plt.ylabel('Metrics')


fpr, tpr, thresholds = roc_curve(obs_heart,pred_prob_heart_qda[:,1])
plt.plot(fpr,tpr)
plt.xlabel('1 - Specificity (false positive rate)')
plt.ylabel('Sensitivity (true positive rate)')


roc_auc_score(obs_heart,pred_prob_heart_qda[:,1])
log_loss(obs_heart, pred_prob_heart_qda[:,1])
brier_score_loss(obs_heart,pred_prob_heart_qda[:,1])


prob_true_qda, prob_pred_qda = calibration_curve(obs_heart,pred_prob_heart_qda[:,1], n_bins=10, strategy = 'quantile')
sm.graphics.abline_plot(intercept = 0,slope = 1,color = 'red')
plt.scatter(prob_true_qda,prob_pred_qda)
plt.xlabel('Observed Probability')
plt.ylabel('Predicted Probability')

# recalibration
plt.hist(pred_prob_heart_qda[:,1])
plt.xlabel('Predicted Probability')
plt.ylabel('Observed Frequencies')

plt.hist(pred_prob_heart);
plt.xlabel('Predicted Probability')
plt.ylabel('Observed Frequencies')

plt.hist(pred_prob_heart);
plt.xlabel('Predicted Probability')
plt.ylabel('Observed Frequencies')


from sklearn.calibration import CalibratedClassifierCV

calibrated_qda_heart = CalibratedClassifierCV(qda_heart)
calibrated_qda_heart_fit = calibrated_qda_heart.fit(X = heart_final.iloc[:,range(0,17)], y = heart_final['heart_disease']) 

pred_prob_heart_qda_calibrated = calibrated_qda_heart_fit.predict_proba(heart_final.iloc[:,range(0,17)])


prob_true_qda_calibrated, prob_pred_qda_calibrated = calibration_curve(obs_heart,pred_prob_heart_qda_calibrated[:,1], n_bins=10, strategy = 'quantile')
sm.graphics.abline_plot(intercept = 0,slope = 1,color = 'red')
plt.scatter(prob_true_qda_calibrated,prob_pred_qda_calibrated)
plt.xlabel('Observed Probability')
plt.ylabel('Predicted Probability')

plt.hist(pred_prob_heart_qda_calibrated[:,1]);
plt.xlabel('Predicted Probability')
plt.ylabel('Observed Frequencies')


roc_auc_score(obs_heart,pred_prob_heart_qda_calibrated[:,1])
log_loss(obs_heart, pred_prob_heart_qda_calibrated[:,1])
brier_score_loss(obs_heart,pred_prob_heart_qda_calibrated[:,1])


# Naive Bayes

# gaussian for continuous predictors
from sklearn.naive_bayes import GaussianNB
gaussNB_heart = GaussianNB()
gaussNB_heart_fit = gaussNB_heart.fit(heart_final[['age','trestbps','chol','thalch','oldpeak']], y = heart_final['heart_disease'])

# categorical predictors
from sklearn.naive_bayes import CategoricalNB
catNB_heart = CategoricalNB()
catNB_heart_fit = catNB_heart.fit(heart_imp_MissForest[['sex','cp','restecg','exang','fbs','slope','thal']], y = heart_final['heart_disease'])

# log P(X_1,Y) and log P(X_2,Y)
joint_logprob_cont_heart =  gaussNB_heart_fit.predict_joint_log_proba(heart_final[['age','trestbps','chol','thalch','oldpeak']])
joint_logprob_cat_heart =  catNB_heart_fit.predict_joint_log_proba(heart_imp_MissForest[['sex','cp','restecg','exang','fbs','slope','thal']])

#  pred prob
pred_prob_heart_nb = np.exp(joint_logprob_cont_heart - catNB_heart_fit.class_log_prior_ + joint_logprob_cat_heart) # compute numerator
pred_prob_heart_nb = np.divide(pred_prob_heart_nb.T,pred_prob_heart_nb.sum(axis = 1)).T # normalize the density


accuracy_scores_nb = np.zeros(41)
balanced_accuracy_scores_nb = np.zeros(41)
f1_scores_nb = np.zeros(41)
f1_scores_neg_nb = np.zeros(41)
matthews_corrcoefs_nb = np.zeros(41)
cohen_kappa_scores_nb = np.zeros(41)

for k in range(41):
    accuracy_thresholds[k] = k/40
    accuracy_scores_nb[k] = accuracy_score(obs_heart,(pred_prob_heart_nb[:,1] > accuracy_thresholds[k]).astype(int))
    balanced_accuracy_scores_nb[k] = balanced_accuracy_score(obs_heart,(pred_prob_heart_nb[:,1] > accuracy_thresholds[k]).astype(int))
    f1_scores_nb[k] = f1_score(obs_heart,(pred_prob_heart_nb[:,1] > accuracy_thresholds[k]).astype(int))
    f1_scores_neg_nb[k] = f1_score(obs_heart,(pred_prob_heart_nb[:,1] > accuracy_thresholds[k]).astype(int),pos_label=0)
    matthews_corrcoefs_nb[k] = matthews_corrcoef(obs_heart,(pred_prob_heart_nb[:,1] > accuracy_thresholds[k]).astype(int))
    cohen_kappa_scores_nb[k] = cohen_kappa_score(obs_heart,(pred_prob_heart_nb[:,1] > accuracy_thresholds[k]).astype(int))
    

plt.plot(accuracy_thresholds,accuracy_scores_nb)
plt.plot(accuracy_thresholds,balanced_accuracy_scores_nb)
plt.plot(accuracy_thresholds,f1_scores_nb)
plt.plot(accuracy_thresholds,f1_scores_neg_nb)
plt.plot(accuracy_thresholds,matthews_corrcoefs_nb)
plt.plot(accuracy_thresholds,cohen_kappa_scores_nb)

plt.legend(['Accuracy','Balanced Accuracy','F1-score','F1-score (for negatives)','Matthews correlation',"Cohen's kappa"], loc="lower right")
plt.xlabel('Probability Threshold')
plt.ylabel('Metrics')


fpr, tpr, thresholds = roc_curve(obs_heart,pred_prob_heart_nb[:,1])
plt.plot(fpr,tpr)
plt.xlabel('1 - Specificity (false positive rate)')
plt.ylabel('Sensitivity (true positive rate)')



roc_auc_score(obs_heart,pred_prob_heart_nb[:,1])
log_loss(obs_heart, pred_prob_heart_nb[:,1])
brier_score_loss(obs_heart,pred_prob_heart_nb[:,1])


prob_true_nb, prob_pred_nb = calibration_curve(obs_heart,pred_prob_heart_nb[:,1], n_bins=10, strategy = 'quantile')
sm.graphics.abline_plot(intercept = 0,slope = 1,color = 'red')
plt.scatter(prob_true_nb,prob_pred_nb)
plt.xlabel('Observed Probability')
plt.ylabel('Predicted Probability')

# recalibration
plt.hist(pred_prob_heart_nb[:,1])

np.random.seed(123)

folds = 5
kf = KFold(n_splits=folds) # create folds

calib_coef =  pd.DataFrame(index=range(folds),columns = ['intercept','slope'])


idx_cv = np.random.choice([*range(len(heart_final))],len(heart_final), replace=False)
for j, (train_index, test_index) in enumerate(kf.split(idx_cv)):
    train_set = idx_cv[train_index]
    test_set = idx_cv[test_index]

    # split the train set
    heart_final_train = heart_final.iloc[train_set]
    heart_final_test = heart_final.iloc[test_set]
        
    heart_imp_MissForest_train = heart_imp_MissForest.iloc[train_set]
    heart_imp_MissForest_test = heart_imp_MissForest.iloc[test_set]

    # learn the naive Bayes classifier
    gaussNB_heart_new = GaussianNB()
    catNB_heart_new = CategoricalNB()
        
    gaussNB_heart_fit_new = gaussNB_heart_new.fit(heart_final_train[['age','trestbps','chol','thalch','oldpeak']], y = heart_final_train['heart_disease'])
    catNB_heart_fit_new = catNB_heart_new.fit(heart_imp_MissForest_train[['sex','cp','restecg','exang','fbs','slope','thal']], y = heart_final_train['heart_disease'])

    # obtain predicted probabilities for the test subset
    joint_logprob_cont_heart_new =  gaussNB_heart_fit_new.predict_joint_log_proba(heart_final_test[['age','trestbps','chol','thalch','oldpeak']])
    joint_logprob_cat_heart_new =  catNB_heart_fit_new.predict_joint_log_proba(heart_imp_MissForest_test[['sex','cp','restecg','exang','fbs','slope','thal']])
    pred_prob_heart_nb_new = np.exp(joint_logprob_cont_heart_new - catNB_heart_fit_new.class_log_prior_ + joint_logprob_cat_heart_new) 
    pred_prob_heart_nb_new = np.divide(pred_prob_heart_nb_new.T,pred_prob_heart_nb_new.sum(axis = 1)).T 

    # fit the logit model on the test subset and extract coefficients
    logitp_nb = np.log(pred_prob_heart_nb_new[:,1]/(1-pred_prob_heart_nb_new[:,1])) # logit p = log p/(1-p)

    calib_model_fit = sm.Logit(endog = heart_final_test['heart_disease'].reset_index(drop = True),exog = pd.DataFrame(logitp_nb).assign(const=1)).fit(disp=0) #logit model
    calib_coef.iloc[j,0] = calib_model_fit.params.iloc[1]
    calib_coef.iloc[j,1] = calib_model_fit.params.iloc[0]
    
calib_coef.mean()

logitp_nb_all = np.log(pred_prob_heart_nb[:,1]/(1-pred_prob_heart_nb[:,1])) # logit p = log p/(1-p)
pred_prob_heart_nb_calib =  1 - 1/(1+np.exp(calib_coef.mean().iloc[1]*logitp_nb_all + calib_coef.mean().iloc[0])) #p_cal = ilogit(slope*log p/(1-p) + intercept)

roc_auc_score(obs_heart,pred_prob_heart_nb_calib)
log_loss(obs_heart, pred_prob_heart_nb_calib)
brier_score_loss(obs_heart,pred_prob_heart_nb_calib)

prob_true_nb_calib, prob_pred_nb_calib = calibration_curve(obs_heart,pred_prob_heart_nb_calib, n_bins=10, strategy = 'quantile')
sm.graphics.abline_plot(intercept = 0,slope = 1,color = 'red')
plt.scatter(prob_true_nb_calib,prob_pred_nb_calib)
plt.xlabel('Observed Probability')
plt.ylabel('Predicted Probability')

plt.hist(pred_prob_heart_nb_calib)


# cross-validation
np.random.seed(123)

rep = 50
folds = 10
kf = KFold(n_splits=10) # create folds

metrics_cv_heart_log_reg =  pd.DataFrame(index=range(rep*folds),columns = ['AUC','Brier score','log score','calibration'])
metrics_cv_heart_lda =  pd.DataFrame(index=range(rep*folds),columns = ['AUC','Brier score','log score','calibration'])
metrics_cv_heart_linear_svm =  pd.DataFrame(index=range(rep*folds),columns = ['AUC','Brier score','log score','calibration'])
metrics_cv_heart_rbf_svm =  pd.DataFrame(index=range(rep*folds),columns = ['AUC','Brier score','log score','calibration'])
metrics_cv_heart_poly_svm =  pd.DataFrame(index=range(rep*folds),columns = ['AUC','Brier score','log score','calibration'])
metrics_cv_heart_sig_svm =  pd.DataFrame(index=range(rep*folds),columns = ['AUC','Brier score','log score','calibration'])
metrics_cv_heart_qda =  pd.DataFrame(index=range(rep*folds),columns = ['AUC','Brier score','log score','calibration'])
metrics_cv_heart_nb =  pd.DataFrame(index=range(rep*folds),columns = ['AUC','Brier score','log score','calibration'])


parameters_svm_linear = {'kernel':['linear'], 'C':[0.001, 0.01, 0.1, 1, 10, 100]}
parameters_svm_rbf = {'kernel':['rbf'], 'C':[0.001, 0.01, 0.1, 1, 10, 100], 'gamma':[0.001, 0.01, 0.1, 1, 10, 100, 1000]}
parameters_svm_poly = {'kernel':['poly'], 'C':[0.001, 0.01, 0.1, 1, 10, 100], 'gamma':[0.001, 0.01, 0.1, 1, 10, 100, 1000],'coef0':[0,1,10], 'degree':[3]}
parameters_svm_sig = {'kernel':['sigmoid'], 'C':[0.001, 0.01, 0.1, 1, 10, 100], 'gamma':[0.001, 0.01, 0.1, 1, 10, 100, 1000],'coef0':[-10, -1, 0.1, 0, 0.1, 1, 10]}

k = 0

for i in range(rep):
    
    idx_cv = np.random.choice([*range(len(heart_final))],len(heart_final), replace=False)
    
    for j, (train_index, test_index) in enumerate(kf.split(idx_cv)):
        
        train_set = idx_cv[train_index]
        test_set = idx_cv[test_index]
        
        heart_train = heart_final.iloc[train_set]
        heart_test = heart_final.iloc[test_set]
        
        heart_imp_train = heart_imp_MissForest.iloc[train_set] # train set for NB
        heart_imp_test = heart_imp_MissForest.iloc[test_set] # train set for NB
        
        
        # logit
        log_reg_new = smf.logit(formula='heart_disease ~ age + sex + cp_typ + cp_atyp + cp_nonang + restecg_hypertrophy + \
                                  restecg_stt + thalch + exang + trestbps + oldpeak + fbs + chol + slope_down + slope_up + \
                                  thal_rev + thal_fixed', data=heart_final)
                                  
        pred_prob_log_reg_new = log_reg_new.fit(disp=0).predict(heart_test)
        logitp_log_reg_new = np.log(pred_prob_log_reg_new/(1-pred_prob_log_reg_new)) 
        
        metrics_cv_heart_log_reg.iloc[k,0] = roc_auc_score(heart_test['heart_disease'],pred_prob_log_reg_new)
        metrics_cv_heart_log_reg.iloc[k,1] = brier_score_loss(heart_test['heart_disease'],pred_prob_log_reg_new)
        metrics_cv_heart_log_reg.iloc[k,2] = log_loss(heart_test['heart_disease'],pred_prob_log_reg_new)
        metrics_cv_heart_log_reg.iloc[k,3] = sm.Logit(endog= heart_test['heart_disease'],exog = pd.DataFrame(logitp_log_reg_new).assign(const=1)).fit(disp=0).params.iloc[0]
        

        # LDA
        lda_new = LinearDiscriminantAnalysis(store_covariance=True)
        lda_new_fit = lda_new.fit(X = heart_train.iloc[:,range(0,17)], y = heart_train['heart_disease']) 
        pred_heart_lda_new = lda_new_fit.predict_proba(heart_test.iloc[:,range(0,17)])[:,1]
        logitp_lda_new = np.log(pred_heart_lda_new/(1-pred_heart_lda_new))
        
        metrics_cv_heart_lda.iloc[k,0] = roc_auc_score(heart_test['heart_disease'],pred_heart_lda_new)
        metrics_cv_heart_lda.iloc[k,1] = brier_score_loss(heart_test['heart_disease'],pred_heart_lda_new)
        metrics_cv_heart_lda.iloc[k,2] = log_loss(heart_test['heart_disease'],pred_heart_lda_new)
        metrics_cv_heart_lda.iloc[k,3] = sm.Logit(endog= heart_test['heart_disease'].reset_index(drop=True),exog = pd.DataFrame(logitp_lda_new).assign(const=1)).fit(disp=0).params.iloc[0]
        
        
        # linear SVM
        svm_linear_pipeline_new = make_pipeline(StandardScaler(),  GridSearchCV(estimator=svm.SVC(),param_grid=parameters_svm_linear, scoring = 'balanced_accuracy', refit=False, n_jobs = 4))
        svm_linear_pipeline_fit_new = svm_linear_pipeline_new.fit(X = heart_train.iloc[:,range(0,17)], y = heart_train['heart_disease'])
        
        svm_linear_params_new = svm_linear_pipeline_fit_new._final_estimator.best_params_
        svm_linear_pipeline_new2 = make_pipeline(StandardScaler(),  svm.SVC(C=svm_linear_params_new['C'], kernel = 'linear', probability = True))
        svm_linear_pipeline_fit_new2 = svm_linear_pipeline_new2.fit(X = heart_train.iloc[:,range(0,17)], y = heart_train['heart_disease'])
        
        pred_prob_linear_svm_new = svm_linear_pipeline_fit_new2.predict_proba(heart_test.iloc[:,range(0,17)])[:,1]
        logitp_linear_svm_new = np.log(pred_prob_linear_svm_new/(1-pred_prob_linear_svm_new))
        
        metrics_cv_heart_linear_svm.iloc[k,0] = roc_auc_score(heart_test['heart_disease'],pred_prob_linear_svm_new)
        metrics_cv_heart_linear_svm.iloc[k,1] = brier_score_loss(heart_test['heart_disease'],pred_prob_linear_svm_new)
        metrics_cv_heart_linear_svm.iloc[k,2] = log_loss(heart_test['heart_disease'],pred_prob_linear_svm_new)
        metrics_cv_heart_linear_svm.iloc[k,3] = sm.Logit(endog= heart_test['heart_disease'].reset_index(drop=True),exog = pd.DataFrame(logitp_linear_svm_new).assign(const=1)).fit(disp=0).params.iloc[0]
        
        
        # rbf SVM
        svm_rbf_pipeline_new = make_pipeline(StandardScaler(),  GridSearchCV(estimator=svm.SVC(),param_grid=parameters_svm_rbf, scoring = 'balanced_accuracy', refit=False, n_jobs = 4))
        svm_rbf_pipeline_fit_new = svm_rbf_pipeline_new.fit(X = heart_train.iloc[:,range(0,17)], y = heart_train['heart_disease'])
        
        svm_rbf_params_new = svm_rbf_pipeline_fit_new._final_estimator.best_params_
        svm_rbf_pipeline_new2 = make_pipeline(StandardScaler(),  svm.SVC(C=svm_rbf_params_new['C'], gamma = svm_rbf_params_new['gamma'], kernel = 'rbf', probability = True))
        svm_rbf_pipeline_fit_new2 = svm_rbf_pipeline_new2.fit(X = heart_train.iloc[:,range(0,17)], y = heart_train['heart_disease'])
        
        pred_prob_rbf_svm_new = svm_rbf_pipeline_fit_new2.predict_proba(heart_test.iloc[:,range(0,17)])[:,1]
        logitp_rbf_svm_new = np.log(pred_prob_rbf_svm_new/(1-pred_prob_rbf_svm_new))
        
        metrics_cv_heart_rbf_svm.iloc[k,0] = roc_auc_score(heart_test['heart_disease'],pred_prob_rbf_svm_new)
        metrics_cv_heart_rbf_svm.iloc[k,1] = brier_score_loss(heart_test['heart_disease'],pred_prob_rbf_svm_new)
        metrics_cv_heart_rbf_svm.iloc[k,2] = log_loss(heart_test['heart_disease'],pred_prob_rbf_svm_new)
        metrics_cv_heart_rbf_svm.iloc[k,3] = sm.Logit(endog= heart_test['heart_disease'].reset_index(drop=True),exog = pd.DataFrame(logitp_rbf_svm_new).assign(const=1)).fit(disp=0).params.iloc[0]
        
        
        # poly SVM
        svm_poly_pipeline_new = make_pipeline(StandardScaler(),  GridSearchCV(estimator=svm.SVC(),param_grid=parameters_svm_poly, scoring = 'balanced_accuracy', refit=False, n_jobs = 4))
        svm_poly_pipeline_fit_new = svm_poly_pipeline_new.fit(X = heart_train.iloc[:,range(0,17)], y = heart_train['heart_disease'])
        
        svm_poly_params_new = svm_poly_pipeline_fit_new._final_estimator.best_params_
        svm_poly_pipeline_new2 = make_pipeline(StandardScaler(),  svm.SVC(C=svm_poly_params_new['C'], gamma = svm_poly_params_new['gamma'], coef0 = svm_poly_params_new['coef0'], kernel = 'poly', degree = 3, probability = True))
        svm_poly_pipeline_fit_new2 = svm_poly_pipeline_new2.fit(X = heart_train.iloc[:,range(0,17)], y = heart_train['heart_disease'])
        
        pred_prob_poly_svm_new = svm_poly_pipeline_fit_new2.predict_proba(heart_test.iloc[:,range(0,17)])[:,1]
        logitp_poly_svm_new = np.log(pred_prob_poly_svm_new/(1-pred_prob_poly_svm_new))
        
        metrics_cv_heart_poly_svm.iloc[k,0] = roc_auc_score(heart_test['heart_disease'],pred_prob_poly_svm_new)
        metrics_cv_heart_poly_svm.iloc[k,1] = brier_score_loss(heart_test['heart_disease'],pred_prob_poly_svm_new)
        metrics_cv_heart_poly_svm.iloc[k,2] = log_loss(heart_test['heart_disease'],pred_prob_poly_svm_new)
        metrics_cv_heart_poly_svm.iloc[k,3] = sm.Logit(endog= heart_test['heart_disease'].reset_index(drop=True),exog = pd.DataFrame(logitp_poly_svm_new).assign(const=1)).fit(disp=0).params.iloc[0]
        
        
        # sig SVM
        svm_sig_pipeline_new = make_pipeline(StandardScaler(),  GridSearchCV(estimator=svm.SVC(),param_grid=parameters_svm_sig, scoring = 'balanced_accuracy', refit=False, n_jobs = 4))
        svm_sig_pipeline_fit_new = svm_sig_pipeline_new.fit(X = heart_train.iloc[:,range(0,17)], y = heart_train['heart_disease'])
        
        svm_sig_params_new = svm_sig_pipeline_fit_new._final_estimator.best_params_
        svm_sig_pipeline_new2 = make_pipeline(StandardScaler(),  svm.SVC(C=svm_sig_params_new['C'], gamma = svm_sig_params_new['gamma'], coef0 = svm_sig_params_new['coef0'], kernel = 'sigmoid', probability = True))
        svm_sig_pipeline_fit_new2 = svm_sig_pipeline_new2.fit(X = heart_train.iloc[:,range(0,17)], y = heart_train['heart_disease'])
        

        pred_prob_sig_svm_new = svm_sig_pipeline_fit_new2.predict_proba(heart_test.iloc[:,range(0,17)])[:,1]
        logitp_sig_svm_new = np.log(pred_prob_sig_svm_new/(1-pred_prob_sig_svm_new))
        
        metrics_cv_heart_sig_svm.iloc[k,0] = roc_auc_score(heart_test['heart_disease'],pred_prob_sig_svm_new)
        metrics_cv_heart_sig_svm.iloc[k,1] = brier_score_loss(heart_test['heart_disease'],pred_prob_sig_svm_new)
        metrics_cv_heart_sig_svm.iloc[k,2] = log_loss(heart_test['heart_disease'],pred_prob_sig_svm_new)
        metrics_cv_heart_sig_svm.iloc[k,3] = sm.Logit(endog= heart_test['heart_disease'].reset_index(drop=True),exog = pd.DataFrame(logitp_sig_svm_new).assign(const=1)).fit(disp=0).params.iloc[0]
        
        
        # QDA
        qda_new = QuadraticDiscriminantAnalysis()
        qda_new_fit = qda_new.fit(X = heart_train.iloc[:,range(0,17)], y = heart_train['heart_disease']) 
        
        calibrated_qda_new = CalibratedClassifierCV(qda_new)
        calibrated_qda_fit_new = calibrated_qda_new.fit(X = heart_train.iloc[:,range(0,17)], y = heart_train['heart_disease']) 
        pred_prob_heart_qda_calibrated = calibrated_qda_fit_new.predict_proba(heart_test.iloc[:,range(0,17)])[:,1]
        
        logitp_qda_new = np.log(pred_prob_heart_qda_calibrated/(1-pred_prob_heart_qda_calibrated))

        metrics_cv_heart_qda.iloc[k,0] = roc_auc_score(heart_test['heart_disease'],pred_prob_heart_qda_calibrated)
        metrics_cv_heart_qda.iloc[k,1] = brier_score_loss(heart_test['heart_disease'],pred_prob_heart_qda_calibrated)
        metrics_cv_heart_qda.iloc[k,2] = log_loss(heart_test['heart_disease'],pred_prob_heart_qda_calibrated)
        metrics_cv_heart_qda.iloc[k,3] = sm.Logit(endog= heart_test['heart_disease'].reset_index(drop=True),exog = pd.DataFrame(logitp_qda_new).assign(const=1)).fit(disp=0).params.iloc[0]
        
    
        # NB    
        folds_inner = 5
        kf_inner = KFold(n_splits=folds_inner) # create folds

        calib_coef_new =  pd.DataFrame(index=range(folds_inner),columns = ['intercept','slope'])

        idx_cv_inner = np.random.choice([*range(len(heart_train))],len(heart_train), replace=False)
        
        for l, (train_index_inner, test_index_inner) in enumerate(kf_inner.split(idx_cv_inner)):
            train_set_inner = idx_cv_inner[train_index_inner]
            test_set_inner = idx_cv_inner[test_index_inner]

            # split the train set
            heart_train_inner = heart_train.iloc[train_set_inner]
            heart_test_inner = heart_train.iloc[test_set_inner]
                
            heart_imp_train_inner = heart_imp_train.iloc[train_set_inner]
            heart_imp_test_inner = heart_imp_train.iloc[test_set_inner]

            # learn the naive Bayes classifier
            gaussNB_new = GaussianNB()
            catNB_new = CategoricalNB()
                
            gaussNB_fit_new = gaussNB_new.fit(heart_train_inner[['age','trestbps','chol','thalch','oldpeak']], y = heart_train_inner['heart_disease'])
            catNBt_fit_new = catNB_new.fit(heart_imp_train_inner[['sex','cp','restecg','exang','fbs','slope','thal']], y = heart_train_inner['heart_disease'])

            # obtain predicted probabilities for the test subset
            joint_logprob_cont_new =  gaussNB_fit_new.predict_joint_log_proba(heart_test_inner[['age','trestbps','chol','thalch','oldpeak']])
            joint_logprob_cat_new =  catNBt_fit_new.predict_joint_log_proba(heart_imp_test_inner[['sex','cp','restecg','exang','fbs','slope','thal']])
            pred_prob_nb_new = np.exp(joint_logprob_cont_new - catNBt_fit_new.class_log_prior_ + joint_logprob_cat_new) 
            pred_prob_nb_new = np.divide(pred_prob_nb_new.T,pred_prob_nb_new.sum(axis = 1)).T 

            # fit the logit model on the test subset and extract coefficients
            logitp_nb_new = np.log(pred_prob_nb_new[:,1]/(1-pred_prob_nb_new[:,1])) # logit p = log p/(1-p)

            calib_model_fit_new = sm.Logit(endog = heart_test_inner['heart_disease'].reset_index(drop = True),exog = pd.DataFrame(logitp_nb_new).assign(const=1)).fit(disp=0) #logit model
            calib_coef_new.iloc[l,0] = calib_model_fit_new.params.iloc[1]
            calib_coef_new.iloc[l,1] = calib_model_fit_new.params.iloc[0]
            
            
        gaussNB_new = GaussianNB()
        catNB_new = CategoricalNB()
            
        gaussNB_fit_new = gaussNB_new.fit(heart_train[['age','trestbps','chol','thalch','oldpeak']], y = heart_train['heart_disease'])
        catNB_fit_new = catNB_new.fit(heart_imp_train[['sex','cp','restecg','exang','fbs','slope','thal']], y = heart_train['heart_disease'])


        joint_logprob_cont_new =  gaussNB_fit_new.predict_joint_log_proba(heart_test[['age','trestbps','chol','thalch','oldpeak']])
        joint_logprob_cat_new =  catNB_fit_new.predict_joint_log_proba(heart_imp_test[['sex','cp','restecg','exang','fbs','slope','thal']])
        pred_prob_nb_new = np.exp(joint_logprob_cont_new - catNB_fit_new.class_log_prior_ + joint_logprob_cat_new)
        pred_prob_nb_new = np.divide(pred_prob_nb_new.T,pred_prob_nb_new.sum(axis = 1)).T # normalize the density
        

        logitp_nb_new = np.log(pred_prob_nb_new[:,1]/(1-pred_prob_nb_new[:,1]))
        pred_prob_nb_new_calib =  1 - 1/(1+np.exp(calib_coef_new.mean().iloc[1]*logitp_nb_new + calib_coef_new.mean().iloc[0]))
        logitp_nb_new_calib = np.log(pred_prob_nb_new_calib/(1-pred_prob_nb_new_calib))
        
        
        metrics_cv_heart_nb.iloc[k,0] = roc_auc_score(heart_test['heart_disease'],pred_prob_nb_new_calib)
        metrics_cv_heart_nb.iloc[k,1] = brier_score_loss(heart_test['heart_disease'],pred_prob_nb_new_calib)
        metrics_cv_heart_nb.iloc[k,2] = log_loss(heart_test['heart_disease'],pred_prob_nb_new_calib)
        metrics_cv_heart_nb.iloc[k,3] = sm.Logit(endog= heart_test['heart_disease'].reset_index(drop=True),exog = pd.DataFrame(logitp_nb_new_calib).assign(const=1)).fit(disp=0).params.iloc[0]


        k = k + 1
       


res = pd.concat([metrics_cv_heart_log_reg.mean(),\
                 metrics_cv_heart_lda.mean(),\
                 metrics_cv_heart_linear_svm.mean(),\
                 metrics_cv_heart_rbf_svm.mean(),\
                 metrics_cv_heart_poly_svm.mean(),\
                 metrics_cv_heart_sig_svm.mean(),\
                 metrics_cv_heart_qda.mean(),\
                 metrics_cv_heart_nb.mean()], axis=1)
    
res.columns = ['logist. reg.','LDA','linear SVM', 'RBF SVM', 'poly SVM', 'sigmoid SVM', 'QDA', 'Naive Bayes']        
res
     