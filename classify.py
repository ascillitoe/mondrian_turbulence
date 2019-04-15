import os

import numpy as np
import pandas as pd

# Load scikit's random forest classifier library and related tools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, multilabel_confusion_matrix, precision_recall_curve

from classify_funcs import *

import matplotlib.pyplot as plt

import warnings
warnings.simplefilter("ignore", FutureWarning)

#print('The scikit-learn version is {}.'.format(sklearn.__version__)) 

#############
# User inputs
#############
casename = 'test'
modelname = 'test1'
data_fileloc = '.'

train_percentage = 0.75
n_estimators = 10
criterion = 'entropy'
random_state = 42

accuracy = True
interpret = False

###########################
# Read in pandas dataframes
###########################
filename = casename + '_q_dataset.csv'
filename = os.path.join(data_fileloc, filename)
print('\nReading pandas features dataset from ', filename)
q_dataset = pd.read_csv(filename)
filename = casename + '_e_dataset.csv'
filename = os.path.join(data_fileloc, filename)
print('\nReading pandas targets dataset from ', filename)
e_dataset = pd.read_csv(filename)

##############
# Prepare data
##############
#TODO - check for missing values and NaN's

# Find feature and target headers
X_headers = q_dataset.columns
Y_headers = e_dataset.columns

nX = X_headers.size
nY = Y_headers.size

#Y_headers = [Y_headers[1]]
#nY = 1 #TODO - temp

print('\nFeatures:')
for i in range(0,nX):
    print('%d/%d: %s' %(i+1,nX,X_headers[i]) )
print('\nTargets:')
for i in range(0,nY):
    print('%d/%d: %s' %(i+1,nY,Y_headers[i]) )

# Split dataset into train and test dataset
print('\nSpliting into test and training data. Training split = ', train_percentage*100, '%')
X_train, X_test, Y_train, Y_test = train_test_split(q_dataset[X_headers], e_dataset[Y_headers],train_size=train_percentage,random_state=random_state)

print('Number of observations for training data = ', X_train.shape[0])
print('Number of observations for test data = ',  X_test.shape[0])

## Scale features NOTE - is this needed? Or is orig non-dim enough?
## Feature Scaling
#print('\nScaling features')
#scaler = StandardScaler()
#X_train = scaler.fit_transform(X_train)
#X_test  = scaler.transform(X_test)
#X_train = pd.DataFrame(X_train,columns=X_headers)
#X_test  = pd.DataFrame( X_test,columns=X_headers)

# Train random forest classifier on training data
print('\nTraining Random Forest classifier on training data...')
clf = RandomForestClassifier(n_estimators = n_estimators, criterion = criterion, random_state = random_state,verbose=True)
clf.fit(X_train, Y_train)
clf.verbose = False #Turn verbose off after this so that accuracy and interpretation prints aren't hidden

############################
# Accuracy metrics and plots
############################
if (accuracy==True):

    # Predict test set for accuracy evaluation
    print('\nUse trained classifier to predict targets based on test features')
    Y_pred = clf.predict(X_test)
    Y_pred = pd.DataFrame(Y_pred,columns=Y_headers)
    
    # Classifier accuracy
    print('\nSubset accuracy of classifier =  %.2f %%' %(accuracy_score(Y_test,Y_pred)*100) )
    print('\nClassifier accuracy for each target:')
    for label in Y_headers:
        print('%s: %.2f %%' %(label, accuracy_score(Y_test[label],Y_pred[label])*100) )
    
    # Classification report (f1-score, recall etc)
    print('\nClassification report for each target:')
    for label in Y_headers:
        print('\nTarget %d: %sG' %(i+1,label))
        print(classification_report(Y_test[label], Y_pred[label],target_names=['Off','On']) )
    
    # Confusion matrix
    confuse_mat = multilabel_confusion_matrix(Y_test, Y_pred)
    print('\nConfusion matrices:')
    i = 0
    for label in Y_headers:
        print('\nTarget %d: %s' %(i+1,label))
        print_cm(confuse_mat[i,:,:], ['Off','On'])
        i += 1
    
    # Plot precision-recall curve
    print('\nPlotting precision-recall score')
    Y_score = clf.predict_proba(X_test)
    for l in range(0,nY):
        if (nY>1):
            precision, recall, thresholds = precision_recall_curve(Y_test[Y_headers[l]], Y_score[l][:,1])
        else:
            precision, recall, thresholds = precision_recall_curve(Y_test[Y_headers[l]], Y_score[:,1])
        plt.step(recall, precision, alpha=0.5,where='post',label=Y_headers[l])
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall curves')
    plt.legend()

####################################
# Interpretability metrics and plots
####################################
if (interpret==True):

    # Calculate feature importances #TODO - how to pick out label from clf to print feature importances and pdp's for specified label
    print('\nCalculating feature importance with eli5')
    from eli5 import explain_weights, format_as_text
    from eli5.sklearn import PermutationImportance
    perm = PermutationImportance(clf, random_state=random_state).fit(X_test, Y_test)
    print(format_as_text(explain_weights(perm, feature_names = X_headers.to_list())))
    
    # Partial dependence plots
    from pdpbox import pdp, get_dataset
    # 1D PDP's
    print('\nPlotting 1D partial dependence plots with pdpbox')
    features_to_plot = ['Pressure/shear stresses','Deviation from parallel shear','Turbulence Re']
    for feature in features_to_plot:
        print('\nPDP plot for ' + feature)
        pdp_dist = pdp.pdp_isolate(model=clf, dataset=X_test, model_features=X_headers, feature=feature)
        pdp.pdp_plot(pdp_dist, feature)
    
    ## 2D PDP's #TODO - pdp contour plots not compatible with current version of matplotlib?
    #print('\nPlotting 2D partial dependence plots with pdpbox')
    #features_to_plot = ['Pressure/shear stresses','Deviation from parallel shear']
    #inter1  =  pdp.pdp_interact(model=clf, dataset=X_test, model_features=X_headers, features=features_to_plot)
    #pdp.pdp_interact_plot(pdp_interact_out=inter1, feature_names=features_to_plot), plot_type='contour')
    
    # SHAP plots
    import shap 
    print('\nSHapley Additive exPlanations (SHAP) plots')
    explainer = shap.TreeExplainer(clf) # Create shap explainer object
    
    # Standard SHAP value plots
    points = [10]
    for point in points:
        print('Plotting SHAP values for point %d' %point)
        data_for_prediction = X_test.iloc[point]
        shap_values = explainer.shap_values(data_for_prediction) # Calculate shap values
        shap.force_plot(explainer.expected_value[1], shap_values[1], data_for_prediction,matplotlib=True,text_rotation=45) #indexed with 1 as plotting for true values (replace with 0 for false)
    
    # Advanced SHAP plots
    data_for_prediction = X_test
    shap_values = explainer.shap_values(data_for_prediction) # Calculate shap values
    
    # SHAP summary plot
    print('\nCreating SHAP summary plot')
    plt.figure()
    shap.summary_plot(shap_values[1],features=X_test,feature_names=X_headers,plot_type='violin')
    #TODO - how to stop waiting before next plot?
    
    # SHAP dependence contribution plots
    print('\nCreating SHAP dependence contribution plots')
    features_to_plot = ['Convection/production of k','Deviation from parallel shear','Pgrad along streamline','Viscosity ratio']
    for feature in features_to_plot:
        print('SHAP dependence plot for ' + feature)
        shap.dependence_plot(feature, shap_values[1],features=X_test,feature_names=X_headers)
    # TODO - need to set interaction_index manually, by first computing SHAP interaction values. automatic selection only approximate?

################
# Save classifer
################
from joblib import dump
filename = modelname + '.joblib'
filename = os.path.join(data_fileloc, filename)
print('\nSaving classifer to ', filename)
dump(clf, filename) 

########
# Finish
########
# Show previously generated plots
plt.show()
