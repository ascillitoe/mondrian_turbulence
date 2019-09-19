import numpy as np
import pandas as pd

import os

import matplotlib.pyplot as plt

def permutation_importance(clf,X,Y,features,random_state=42,scoring=None):
    from eli5 import explain_weights, format_as_text
    from eli5.sklearn import PermutationImportance
    
    # Extract the classifier object from the clf multilearn object
    clf.verbose = False #Turn verbose off after this to tidy prints

    # Calculate feature importances #TODO - how to pick out label from clf to print feature importances and pdp's for specified label
    perm = PermutationImportance(clf, random_state=random_state,scoring=scoring).fit(X, Y)
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

def SHAP_values(clf,X):
    import shap 

    print('\nFinding Shapley values')

    clf.verbose = False #Turn verbose off after this to tidy prints

    explainer = shap.TreeExplainer(clf) # Create shap explainer object
    shap_values = explainer.shap_values(X) # Calculate shap values
    
    clf.verbose = True

    print('Finished finding Shapley values')

    return shap_values


def SHAP_summary(X,feature_names,shap_values):
    import shap

    print('Creating SHAP summary plot')

    # SHAP summary plot
    plt.figure()
    shap.summary_plot(shap_values[1],features=X,feature_names=feature_names,plot_type='violin',show=False)

def SHAP_DepenContrib(X,feature_names,feature,shap_values,interact='auto'):
    import shap

    print('Creating SHAP dependence contribution plot')

    # SHAP dependence contribution plots
#    plt.figure()
    shap.dependence_plot(feature, shap_values[1],features=X,feature_names=feature_names,show=False,interaction_index=interact)

def SHAP_inter_values(clf,X):
    import shap

    print('\nFinding Shapley interaction values')

    clf.verbose = False #Turn verbose off after this to tidy prints

    explainer = shap.TreeExplainer(clf) # Create shap explainer object
    shap_inter_values = explainer.shap_interaction_values(X) # Calculate shap interaction values

    print('Finished finding Shapley interaction values')

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

def SHAP_force(clf,data,point,type='bar',index=None): 
    import shap

    print('\n SHAP force plot')

    if (index is None):
        X      = data.pd
        points = data.vtk.points
    else:
        X      = data.pd.iloc[index]  #index is array to index data with. i.e. if only sampling from test/train data
        points = data.vtk.points[index]

    clf.verbose = False #Turn verbose off after this to tidy prints
    explainer = shap.TreeExplainer(clf)

    # Find the closest datapoint to "point", and create SHAP explainer for that datapoint
    dist = points-point
    dist = np.sqrt(dist[:,0]**2.0 + dist[:,1]**2.0 + dist[:,2]**2.0)
    loc = np.argmin(dist)

    print('point = ', point)
    print('nearest point = ',points[loc,:])
    print('distance = ',dist[loc])

    datapoint = X.iloc[loc]
    print(datapoint)
    shap_value = explainer.shap_values(datapoint) # Calculate shap values

    clf.verbose = True

    if(type=='force'):
        shap.force_plot(explainer.expected_value[1], shap_value[1], datapoint,matplotlib=True,text_rotation=45,show=False) #indexed with 1 as plotting for true values (replace with 0 for false)
    elif(type=='bar'):
        fs = 18
        import matplotlib 
        matplotlib.rc('xtick', labelsize=20) 
        matplotlib.rc('ytick', labelsize=20)
        sort_ind = np.argsort(shap_value[1])[::-1]
        y_pos = np.arange(np.size(shap_value[1]))
        plt.figure()
        plt.barh(y_pos,shap_value[1][sort_ind],align='center')
        yprob = clf.predict_proba(datapoint.to_numpy().reshape(1,-1))[0,1]
        plt.title('Classifier probability = %.2f' %(yprob),fontsize=fs)
        ax = plt.gca()
        # Set ytick labels
        feature_names = X.columns[sort_ind]
        yticklabel = ['%s = %.2f' %(feature,datapoint[feature]) for feature in feature_names]
        ax.set_yticks(y_pos)
        ax.set_yticklabels(yticklabel,fontsize=fs)
        ax.set_xlabel('SHAP value',fontsize=fs)


def viz_tree(clf,Xdata,Ydata,label,outfile,point=None):
    from dtreeviz.trees import dtreeviz
    from sklearn.tree import export_graphviz

    X      = Xdata.pd
    Y      = Ydata.pd
    X_headers = X.columns
    Y_headers = Y.columns

    # If "point" given as argument, find the closest datapoint to "point", will do treewalk for this observation
    if(point is not None):
        points = Xdata.vtk.points
        dist = points-point
        dist = np.sqrt(dist[:,0]**2.0 + dist[:,1]**2.0 + dist[:,2]**2.0)
        loc = np.argmin(dist)

        print('point = ', point)
        print('nearest point = ',points[loc,:])
        print('distance = ',dist[loc])

        datapoint = X.iloc[loc]
    else:
        datapoint = None

    # Extract the classifier object from the clf multilearn object
    index = Y_headers.to_list().index(label)
    clf = clf.classifiers_[index]

    # TODO: check if clf is a decision tree

    viz = dtreeviz(clf, X, Y[label],
              feature_names=X_headers,
              target_name=label,class_names=["False","True"],X=datapoint)

    viz.save(outfile)

def viz_tree_simple(clf,label,features,labels,outfile):
    import graphviz
    from sklearn.tree import export_graphviz

    # Extract the classifier object from the clf multilearn object
    index = labels.to_list().index(label)
    clf = clf.classifiers_[index]

    # TODO: check if clf is a decision tree

    dot_data = export_graphviz(clf,feature_names=features,filled=True,rounded=True,special_characters=True,proportion=True)
    graph = graphviz.Source(dot_data)
    graph.render(outfile)

def decision_surface(clf,Xdata,Ydata,label,feature_pair,point,valrange):
    from mlxtend.plotting import plot_decision_regions

    print('\nPlotting decision surface for ' + feature_pair[0] + ' and ' + feature_pair[1])

    X = Xdata.pd
    Y = Ydata.pd
    X_headers = X.columns
    Y_headers = Y.columns

    # Extract the classifier object from the clf multilearn object
    index = Y_headers.to_list().index(label)
    clf = clf.classifiers_[index]
    clf.verbose = False
    
    # Get indexes of X_headers that correspond to the features in feature_pair
    features = [index for index, item in enumerate(X_headers.to_list()) if item in feature_pair]

    # Find nearest point to "point". Feature values from this index will be used to set filler_feature_values for later
    points = Xdata.vtk.points
    dist = points-point
    dist = np.sqrt(dist[:,0]**2.0 + dist[:,1]**2.0 + dist[:,2]**2.0)
    loc = np.argmin(dist)

    print('point = ', point)
    print('nearest point = ',points[loc,:])
    print('distance = ',dist[loc])

    datapoint = X.iloc[loc]

    # Create dict objects containing values (and ranges) for all features NOT in feature_pair
    keys = [index for index, item in enumerate(X_headers.to_list()) if item not in feature_pair]
    values = X[X_headers[keys]]
    ranges = np.maximum(1e-12,np.abs(np.max(values,axis=0)-np.min(values,axis=0))*valrange)
    values = values.iloc[loc]
    filler_feature_values = {k: v for k, v in zip(keys, values)}
    filler_feature_ranges = {k: v for k, v in zip(keys, ranges)}
    
    print('values of other features:\n', filler_feature_values)
    print('ranges of other features:\n', filler_feature_ranges)

    # Plotting decision regions
    fig, ax = plt.subplots()

    X = X.to_numpy()
    Y = Y[label].to_numpy()
    plot_decision_regions(X, Y, clf=clf, feature_index=features,
                          filler_feature_values=filler_feature_values,filler_feature_ranges=filler_feature_ranges,
                          legend=0, ax=ax)
    ax.set_xlabel(X_headers[features[0]]) 
    ax.set_ylabel(X_headers[features[1]]) 

    ax.set_xlim([np.min(X[:,features[0]]),np.max(X[:,features[0]])])
    ax.set_ylim([np.min(X[:,features[1]]),np.max(X[:,features[1]])])
    
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, ['False', 'True'])
    #ax.set_title('Feature 3 = {}'.format(value))
    
