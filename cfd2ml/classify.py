import numpy as np
import pandas as pd

import os

from joblib import dump, load
import matplotlib.pyplot as plt

from skmultilearn.problem_transform import BinaryRelevance

def RF_classifier(q_dataset,e_dataset,modelname,saveloc=None,accuracy=False,train_percentage=0.75,n_estimators=10,criterion='entropy',random_state=42):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    if(accuracy==True): 
        from sklearn.metrics import classification_report, multilabel_confusion_matrix, precision_recall_curve
        from cfd2ml.utilities import print_cm

    ##############
    # Prepare data
    ##############
    #TODO - check for missing values and NaN's
    
    # Find feature and target headers
    X_headers = q_dataset.columns
    Y_headers = e_dataset.columns
    
    nX = X_headers.size
    nY = Y_headers.size
    
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
    clf = BinaryRelevance(classifier=clf,require_dense=[False,True])
    clf.fit(X_train, Y_train)

    # Accuracy metrics and plots
    ############################
    clf.verbose = False #Turn verbose off after this to tidy prints

    # Predict test set for accuracy evaluation
    print('\nUse trained classifier to predict targets based on test features')
    Y_pred = clf.predict(X_test)
    Y_pred = pd.DataFrame(Y_pred.toarray(),columns=Y_headers)
    
    # Classifier accuracy
    print('\nSubset accuracy of classifier =  %.2f %%' %(accuracy_score(Y_test,Y_pred)*100) )
    print('\nClassifier accuracy for each target:')
    for label in Y_headers:
        print('%s: %.2f %%' %(label, accuracy_score(Y_test[label],Y_pred[label])*100) )

    if (accuracy==True):
        # Classification report (f1-score, recall etc)
        print('\nClassification report for each target:')
        i = 0
        for label in Y_headers:
            print('\nTarget %d: %s' %(i+1,label))
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
        Y_score = pd.DataFrame(Y_score.toarray(),columns=Y_headers)

        for label in Y_headers:
            precision, recall, thresholds = precision_recall_curve(Y_test[label], Y_score[label])
            plt.step(recall, precision, alpha=0.5,where='post',label=label)
        
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

    return clf, X_train.index, X_test.index


def permutation_importance(clf,X,Y,features,label,random_state=42):
    from eli5 import explain_weights, format_as_text
    from eli5.sklearn import PermutationImportance
    
    # Extract the classifier object from the clf multilearn object
    index = Y.columns.to_list().index(label)
    clf = clf.classifiers_[index]
    clf.verbose = False #Turn verbose off after this to tidy prints
    Y = Y[label]

    # Calculate feature importances #TODO - how to pick out label from clf to print feature importances and pdp's for specified label
    perm = PermutationImportance(clf, random_state=random_state).fit(X, Y)
    print(format_as_text(explain_weights(perm, feature_names = features),show=['feature_importances']))

    clf.verbose = True # reset

    return
  
def pdp_1d(clf,X,Y,features_to_plot,label,detailed=False):
    from pdpbox import pdp

    figs = list()
    axs  = list()

    # Extract the classifier object from the clf multilearn object
    index = Y.columns.to_list().index(label)
    clf = clf.classifiers_[index]
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

def pdp_2d(clf,X,Y,features_to_plot,label,plot_type='contour'):
    from pdpbox import pdp

    # Extract the classifier object from the clf multilearn object
    index = Y.columns.to_list().index(label)
    clf = clf.classifiers_[index]
    clf.verbose = False #Turn verbose off after this to tidy prints

    inter  =  pdp.pdp_interact(model=clf, dataset=X, model_features=X.columns.to_list(), features=features_to_plot,percentile_ranges=[(5,95),(5,95)])
    if(plot_type=='grid'):
        fig, ax = pdp.pdp_interact_plot(pdp_interact_out=inter, feature_names=features_to_plot, plot_type='grid',
                x_quantile=True,plot_pdp=True)
    elif(plot_type=='contour'):
        fig, ax = pdp.pdp_interact_plot(pdp_interact_out=inter, feature_names=features_to_plot, plot_type='contour')

    clf.verbose = True # reset

    return fig, ax

def target_plot(X,Y,features_to_plot,label,grid_range=None):
    from pdpbox import info_plots

    figs = list()
    axs  = list()

    df = pd.concat([X, Y], axis=1, join_axes=[X.index])
    for feature in features_to_plot:
        if(grid_range is None):
            fig, ax, summary_df = info_plots.target_plot(df,feature=feature,feature_name=feature,target=label,grid_type='equal')
        else:
            fig, ax, summary_df = info_plots.target_plot(df,feature=feature,feature_name=feature,target=label,grid_type='equal',
                    show_outliers='True',grid_range=grid_range)
        figs.append(fig)
        axs.append(ax)

    return figs, axs

def target_plot_inter(X,Y,features,label,grid_ranges=None):
    from pdpbox import info_plots

    df = pd.concat([X, Y], axis=1, join_axes=[X.index])

    if(grid_ranges is None):
        fig, ax, summary_df = info_plots.target_plot_interact(df,features=features,feature_names=features,target=label,grid_types=['equal','equal'])
    else:
        fig, ax, summary_df = info_plots.target_plot_interact(df,features=features,feature_names=features,target=label,grid_types=['equal','equal'],
                 show_outliers='True',grid_ranges=grid_ranges)

    return fig, ax

def pred_target_plot(clf,X,Y,features_to_plot,label,grid_range=None):
    from pdpbox import info_plots
    
    figs = list()
    axs  = list()

    pd.options.mode.chained_assignment = None # Turn warning msg off
    # Extract the classifier object from the clf multilearn object
    index = Y.columns.to_list().index(label)
    clf = clf.classifiers_[index]
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

def pred_target_plot_inter(clf,X,Y,features,label,grid_ranges=None):
    from pdpbox import info_plots
    
    pd.options.mode.chained_assignment = None # Turn warning msg off

    # Extract the classifier object from the clf multilearn object
    index = Y.columns.to_list().index(label)
    clf = clf.classifiers_[index]
    clf.verbose = False #Turn verbose off after this to tidy prints

    if(grid_ranges is None):
        fig, ax, summary_df = info_plots.actual_plot_interact(clf,X,features=features,feature_names=features,grid_types=['equal','equal'])
    else:
        fig, ax, summary_df = info_plots.actual_plot_interact(clf,X,features=features,feature_names=features,grid_types=['equal','equal'],
                 show_outliers='True',grid_ranges=grid_ranges)

    pd.options.mode.chained_assignment = 'warn' # Turn warning msg back on
    clf.verbose = True # reset

    return fig, ax

def SHAP_values(clf,X,labels,label):
    import shap 

    print('\nFinding Shapley values for ' + label)

    # Extract the classifier object from the clf multilearn object
    index = labels.to_list().index(label)
    clf = clf.classifiers_[index]

    clf.verbose = False #Turn verbose off after this to tidy prints

    explainer = shap.TreeExplainer(clf) # Create shap explainer object
    shap_values = explainer.shap_values(X) # Calculate shap values
    
    clf.verbose = True

    print('FInished finding Shapley values for ' + label)

    return shap_values


def SHAP_summary(X,feature_names,shap_values=None,clf=None,labels=None,label=None):
    import shap

    print('Creating SHAP summary plot')

    if(shap_values is None):
        if(labels is None or label is None or clf is None):
            quit('Stopping - Must provide clf, labels and label arguments to SHAP_summary if shap_values not provided')
        shap_values = SHAP_values(clf,X,labels,label)

    # SHAP summary plot
    plt.figure()
    shap.summary_plot(shap_values[1],features=X,feature_names=feature_names,plot_type='violin',show=False)
    
def SHAP_DepenContrib(X,feature_names,feature,shap_values=None,clf=None,labels=None,label=None,interact='auto'):
    import shap

    print('Creating SHAP dependence contribution plot')

    if(shap_values is None):
        if(labels is None or label is None or clf is None):
            quit('Stopping - Must provide clf, labels and label arguments to SHAP_summary if shap_values not provided')
        shap_values = SHAP_values(clf,X,labels,label)

    # SHAP dependence contribution plots
    plt.figure()
    shap.dependence_plot(feature, shap_values[1],features=X,feature_names=feature_names,show=False,interaction_index=interact)

def SHAP_inter_values(clf,X,labels,label):
    import shap

    print('\nFinding Shapley interaction values for ' + label)

    # Extract the classifier object from the clf multilearn object
    index = labels.to_list().index(label)
    clf = clf.classifiers_[index]
    clf.verbose = False #Turn verbose off after this to tidy prints

    explainer = shap.TreeExplainer(clf) # Create shap explainer object
    shap_inter_values = explainer.shap_interaction_values(X) # Calculate shap interaction values

    print('FInished finding Shapley interaction values for ' + label)

    return shap_inter_values

def SHAP_inter_grid(shap_inter_values,feature_names):
    print('\n Creating Shapley interaction values matrix')

    shap_inter_values = shap_inter_values[1]
    tmp = np.abs(shap_inter_values).sum(0)
    for i in range(tmp.shape[0]):
        tmp[i,i] = 0
    inds = np.argsort(-tmp.sum(0))[:50]
    tmp2 = tmp[inds,:][:,inds]
    plt.figure(figsize=(12,12))
    plt.imshow(tmp2)
    plt.yticks(range(tmp2.shape[0]), feature_names[inds], rotation=50.4, horizontalalignment="right")
    plt.xticks(range(tmp2.shape[0]), feature_names[inds], rotation=50.4, horizontalalignment="left")
    plt.gca().xaxis.tick_top()

def SHAP_force(clf,data,index,point,labels,label,type='bar'):
    import shap

    print('\n SHAP force plot')

    X      = data.pd.iloc[index]
    points = data.vtk.points[index]

    dist = points-point
    dist = np.sqrt(dist[:,0]**2.0 + dist[:,1]**2.0 + dist[:,2]**2.0)
    loc = np.argmin(dist)

    print('point = ', point)
    print('nearest point = ',points[loc,:])
    print('distance = ',dist[loc])

    # Extract the classifier object from the clf multilearn object
    index = labels.to_list().index(label)
    clf = clf.classifiers_[index]
    clf.verbose = False #Turn verbose off after this to tidy prints
    explainer = shap.TreeExplainer(clf)

    datapoint = X.iloc[loc]
    shap_value = explainer.shap_values(datapoint) # Calculate shap values
    clf.verbose = True

    if(type=='force'):
        shap.force_plot(explainer.expected_value[1], shap_value[1], datapoint,matplotlib=True,text_rotation=45,show=False) #indexed with 1 as plotting for true values (replace with 0 for false)
    elif(type=='bar'):
        sort_ind = np.argsort(shap_value[1])[::-1]
        y_pos = np.arange(np.size(shap_value[1]))
        plt.barh(y_pos,shap_value[1][sort_ind],align='center')
        ax = plt.gca()
        ax.set_yticks(y_pos)
        ax.set_yticklabels(X.columns[sort_ind])
        ax.set_xlabel('SHAP value')
