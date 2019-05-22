import numpy as np
import pandas as pd

import os

import matplotlib.pyplot as plt

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

def target_plot(X,Y,features_to_plot,labels,grid_range=None):
    from pdpbox import info_plots

    figs = list()
    axs  = list()

    df = pd.concat([X, Y], axis=1, join_axes=[X.index])
    for feature in features_to_plot:
        if(grid_range is None):
            fig, ax, summary_df = info_plots.target_plot(df,feature=feature,feature_name=feature,target=labels,grid_type='equal')
        else:
            fig, ax, summary_df = info_plots.target_plot(df,feature=feature,feature_name=feature,target=labels,grid_type='equal',
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
        plt.figure()
        plt.barh(y_pos,shap_value[1][sort_ind],align='center')
        ax = plt.gca()
        ax.set_yticks(y_pos)
        ax.set_yticklabels(X.columns[sort_ind])
        ax.set_xlabel('SHAP value')
