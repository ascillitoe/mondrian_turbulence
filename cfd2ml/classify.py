import numpy as np
import pandas as pd

from joblib import dump
import os

from cfd2ml.utilities import print_cm

import matplotlib.pyplot as plt

def RF_classifier(q_dataset,e_dataset,casename,modelname,saveloc=None,accuracy=False,train_percentage=0.75,n_estimators=10,criterion='entropy',random_state=42):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    if(accuracy==True): from sklearn.metrics import classification_report, multilabel_confusion_matrix, precision_recall_curve

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

    # Accuracy metrics and plots
    ############################
    clf.verbose = False #Turn verbose off after this to tidy prints

    # Predict test set for accuracy evaluation
    print('\nUse trained classifier to predict targets based on test features')
    Y_pred = clf.predict(X_test)
    Y_pred = pd.DataFrame(Y_pred,columns=Y_headers)
    
    # Classifier accuracy
    print('\nSubset accuracy of classifier =  %.2f %%' %(accuracy_score(Y_test,Y_pred)*100) )
    print('\nClassifier accuracy for each target:')
    for label in Y_headers:
        print('%s: %.2f %%' %(label, accuracy_score(Y_test[label],Y_pred[label])*100) )

    if (accuracy==True):
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
        plt.show()

    # Save classifier
    #################
    if (saveloc is not None):
        filename = modelname + '.joblib'
        filename = os.path.join(saveloc, filename)
        print('\nSaving classifer to ', filename)
        dump(clf, filename) 

    clf.verbose = True # reset

    return clf, X_test, Y_test


def permutation_importance(clf,X,Y,features,random_state=42):
    from eli5 import explain_weights, format_as_text
    from eli5.sklearn import PermutationImportance

    clf.verbose = False #Turn verbose off after this to tidy prints

    # Calculate feature importances #TODO - how to pick out label from clf to print feature importances and pdp's for specified label
    perm = PermutationImportance(clf, random_state=random_state).fit(X, Y)
    print(format_as_text(explain_weights(perm, feature_names = features)))

    clf.verbose = True # reset

    return
  
def pdp_1d(clf,X,features_to_plot,detailed=False):
    from pdpbox import pdp

    figs = list()
    axs  = list()

    clf.verbose = False #Turn verbose off after this to tidy prints

    for feature in features_to_plot:
        pdp_dist = pdp.pdp_isolate(model=clf, dataset=X, model_features=X.columns.to_list(), feature=feature)
        if(detailed==True):
            fig, ax = pdp.pdp_plot(pdp_dist, feature,
                    plot_pts_dist=True,cluster=True,n_cluster_centers=50,x_quantile=True,show_percentile=True)
        else:
            fig, ax = pdp.pdp_plot(pdp_dist, feature)

        figs.append(fig)
        axs.append(ax)

    clf.verbose = True # reset

    return figs, axs

def pdp_2d(clf,X,features_to_plot,plot_type='contour'):
    from pdpbox import pdp

    clf.verbose = False #Turn verbose off after this to tidy prints

    inter  =  pdp.pdp_interact(model=clf, dataset=X, model_features=X.columns.to_list(), features=features_to_plot,percentile_ranges=[(5,95),(5,95)])
    if(plot_type=='grid'):
        fig, ax = pdp.pdp_interact_plot(pdp_interact_out=inter, feature_names=features_to_plot, plot_type='grid',
                x_quantile=True,plot_pdp=True)
    elif(plot_type=='contour'):
        fig, ax = pdp.pdp_interact_plot(pdp_interact_out=inter, feature_names=features_to_plot, plot_type='contour')

    clf.verbose = True # reset

    return fig, ax

def target_plot(X,Y,features_to_plot,target,grid_range=None):
    from pdpbox import info_plots

    figs = list()
    axs  = list()

    df = pd.concat([X, Y], axis=1, join_axes=[X.index])
    for feature in features_to_plot:
        if(grid_range is None):
            fig, ax, summary_df = info_plots.target_plot(df,feature=feature,feature_name=feature,target=target,grid_type='equal')
        else:
            fig, ax, summary_df = info_plots.target_plot(df,feature=feature,feature_name=feature,target=target,grid_type='equal',
                    show_outliers='True',grid_range=grid_range)
        figs.append(fig)
        axs.append(ax)

    return figs, axs

def target_plot_inter(X,Y,features,target,grid_ranges=None):
    from pdpbox import info_plots

    df = pd.concat([X, Y], axis=1, join_axes=[X.index])

    if(grid_ranges is None):
        fig, ax, summary_df = info_plots.target_plot_interact(df,features=features,feature_names=features,target=target,grid_types=['equal','equal'])
    else:
        fig, ax, summary_df = info_plots.target_plot_interact(df,features=features,feature_names=features,target=target,grid_types=['equal','equal'],
                 show_outliers='True',grid_ranges=grid_ranges)

    return fig, ax

def pred_target_plot(clf,X,features_to_plot,grid_range=None):
    from pdpbox import info_plots
    
    figs = list()
    axs  = list()

    pd.options.mode.chained_assignment = None # Turn warning msg off
    clf.verbose = False #Turn verbose off after this to tidy prints

    for feature in features_to_plot:
        if(grid_range is None):
            fig, ax, summary_df = info_plots.actual_plot(clf,X,feature=feature,feature_name=feature,grid_type='equal')
        else:
            fig, ax, summary_df = info_plots.actual_plot(clf,X,feature=feature,feature_name=feature,grid_type='equal',
                    show_outliers='True',grid_range=grid_range)
        figs.append(fig)
        axs.append(ax)

    pd.options.mode.chained_assignment = 'warn' # Turn warning msg back on
    clf.verbose = True # reset

    return figs, axs

def pred_target_plot_inter(clf,X,features,grid_ranges=None):
    from pdpbox import info_plots
    
    pd.options.mode.chained_assignment = None # Turn warning msg off
    clf.verbose = False #Turn verbose off after this to tidy prints

    if(grid_ranges is None):
        fig, ax, summary_df = info_plots.actual_plot_interact(clf,X,features=features,feature_names=features,grid_types=['equal','equal'])
    else:
        fig, ax, summary_df = info_plots.actual_plot_interact(clf,X,features=features,feature_names=features,grid_types=['equal','equal'],
                 show_outliers='True',grid_ranges=grid_ranges)

    pd.options.mode.chained_assignment = 'warn' # Turn warning msg back on
    clf.verbose = True # reset

    return fig, ax

#def SHAP_ini(clf,X_test)
#    import shap 
#    explainer = shap.TreeExplainer(clf) # Create shap explainer object
#    
#    # Standard SHAP value plots
#    for point in points:
#        print('Plotting SHAP values for point %d' %point)
#        data_for_prediction = X_test.iloc[point]
#        shap_values = explainer.shap_values(data_for_prediction) # Calculate shap values
#        shap.force_plot(explainer.expected_value[1], shap_values[1], data_for_prediction,matplotlib=True,text_rotation=45) #indexed with 1 as plotting for true values (replace with 0 for false)
#    
#    # Advanced SHAP plots
#    data_for_prediction = X_test
#    shap_values = explainer.shap_values(data_for_prediction) # Calculate shap values
#    
#    # SHAP summary plot
#    print('\nCreating SHAP summary plot')
#    plt.figure()
#    shap.summary_plot(shap_values[1],features=X_test,feature_names=X_headers,plot_type='violin')
#    #TODO - how to stop waiting before next plot?
#    
#    # SHAP dependence contribution plots
#    print('\nCreating SHAP dependence contribution plots')
#    features_to_plot = ['Convection/production of k','Deviation from parallel shear','Pgrad along streamline','Viscosity ratio']
#    for feature in features_to_plot:
#        print('SHAP dependence plot for ' + feature)
#        shap.dependence_plot(feature, shap_values[1],features=X_test,feature_names=X_headers)
#    # TODO - need to set interaction_index manually, by first computing SHAP interaction values. automatic selection only approximate?
#
#
#########
## Finish
#########
## Show previously generated plots
#plt.show()
