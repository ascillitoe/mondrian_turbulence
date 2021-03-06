import numpy as np
import pandas as pd

import os

from joblib import dump, load
import pickle
from cfd2ml.base import CaseData

def regress(json):
    from cfd2ml.regress import RF_regressor
    from sklearn.feature_selection import VarianceThreshold

    print('\n-----------------------')
    print('Started training')
    print('Type: Regression')
    print('-----------------------')

    modelname  = json['save_model']
    datloc     = json['training_data_location']
    cases      = json['training_data_cases']
    target     = json['target']

    options = None
    sample  = None
    if (("options" in json)==True):
        options    = json['options']
        if(("sample" in options)==True):
            sample = options['sample']
        if(("features_to_drop" in options)==True):
            features_to_drop = options['features_to_drop']
        else:
            features_to_drop = None
        if(("features_to_keep" in options)==True):
            features_to_keep = options['features_to_keep']
        else:
            features_to_keep = None
        if(features_to_drop and features_to_keep): quit('features_to_drop and features_to_keep both set')

    # Read data
    X_data = pd.DataFrame()
    Y_data = pd.DataFrame()

    # Read in each data set, append into single X and Y df's to pass to regressor. 
    # "group" identifier column is added to X_case to id what case the data came from. 
    # (Used in LeaveOneGroupOut CV later)
    caseno = 1
    for case in cases:
        # Read in RANS (X) data
        filename = os.path.join(datloc,case+'_X.pkl')
        X_case = CaseData(filename)

        # Add "group" id column
        X_case.pd['group'] = caseno
        caseno += 1

        # Read in HiFi (Y) data
        filename = os.path.join(datloc,case+'_Y.pkl')
        Y_case = CaseData(filename)

        # Add X and Y data to df's
        X_data = X_data.append(X_case.pd,ignore_index=True)
        Y_data = Y_data.append(Y_case.pd,ignore_index=True)

    nrows = len(X_data.index)
    print('Original number of rows in dataset: = ', nrows)

    # Remove duplicate observations
    index = X_data.round(3).drop_duplicates().index
    X_data = X_data.iloc[index].reset_index(drop=True)
    Y_data = Y_data.iloc[index].reset_index(drop=True)
    nrows = len(X_data.index)
    print('Number of rows in dataset after removing duplicates: = ', nrows)

    # Randomly sample a % of the data
    if(sample is not None):
        index = np.random.choice(X_data.index,int(sample*nrows))
        X_data = X_data.iloc[index].reset_index(drop=True) 
        Y_data = Y_data.iloc[index].reset_index(drop=True)
        nrows = len(X_data.index)
        print('Number of rows in dataset after sampling: = ', nrows)

    # Write final combined data from all cases to file
    X_data.to_csv(modelname + '_Xdat.csv',index=False)
    Y_data.to_csv(modelname + '_Ydat.csv',index=False)

    if (features_to_drop is not None): 
        X_data = X_data.drop(columns=features_to_drop)
    elif (features_to_keep is not None):
        features_to_keep.append('group')
        X_data = X_data[features_to_keep]

    # Train regressor
    rf_regr =  RF_regressor(X_data,Y_data[target],options=options) 

    # Save regressor
#    filename = modelname + '.joblib'
    filename = modelname + '.p'
    print('\nSaving regressor to ', filename)
#    if mondrian:
    pickle.dump(rf_regr, open(filename, 'wb')) #joblib not working for mondrian forest for some reason...
#    else:
#        dump(rf_regr, filename, protocol=2) 

    print('\n-----------------------')
    print('Finished training')
    print('-----------------------')

def RF_regressor(X_data,Y_data,options=None):
    from sklearn.ensemble import RandomForestRegressor

    ####################
    # Parse user options
    ####################
    params = {}
    gridsearch   = False
    GS_settings  = None
    randomsearch = False
    RS_settings  = None
    feature_selection = False
    accuracy = False
    cv_type = 'logo'
    scoring = 'neg_mean_absolute_error'
    mondrian = False
    search_std = False

    if (options is not None):

        if (("RF_parameters" in options)==True):
            params = options['RF_parameters']

        if (("grid_search" in options)==True):
            from sklearn.model_selection import GridSearchCV
            import time
            gridsearch = True
            GS_params   = options['grid_search']['parameter_grid']
            if (("settings" in options['grid_search'])==True): GS_settings = options['grid_search']['settings']
            if (("search std" in options['grid_search'])==True):
                search_std = options['grid_search']['search std']

        if (("random_search" in options)==True):
            from sklearn.model_selection import RandomizedSearchCV
            from cfd2ml.utilities import convert_param_dist
            import time
            randomsearch = True
            RS_params, RS_Nmax   = convert_param_dist(options['random_search']['parameter_grid'])
            print('RS_Nmax = ', RS_Nmax)
            if (("settings" in options['random_search'])==True): RS_settings = options['random_search']['settings'] 

        if(randomsearch==True and gridsearch==True): quit('********** Stopping! grid_search and random_search both set *********')

        if (("feature_selection" in options)==True):
            from cfd2ml.utilities import RFE_perm
            feature_selection = True
            feats =  options['feature_selection']['feats']
#            if("step"         in options['feature_selection']): step         = options['feature_selection']['step']
#            if("min_features" in options['feature_selection']): min_features = options['feature_selection']['min_features']
            if(randomsearch==True or gridsearch==True): quit('******** Stopping! grid/random_search and feature selection both set ********')

        if (("accuracy" in options)==True):
            accuracy = options['accuracy']
            if (accuracy==True):
                from sklearn.model_selection import cross_validate
                from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

        if (("scoring" in options)==True):
            scoring = options['scoring']

        if (("cv_type" in options)==True):
            cv_type = options['cv_type']

        if (("mondrian" in options)==True):
            mondrian = options['mondrian']
            if mondrian: from skgarden import MondrianForestRegressor

    ##############
    # Prepare data
    ##############
    if(cv_type=='logo'): groups = X_data['group']
    X_data = X_data.drop(columns='group')

    # Find feature and target headers
    X_headers = X_data.columns
    Y_header  = Y_data.name

    nX = X_headers.size
    print('\nFeatures:')
    for i in range(0,nX):
        print('%d/%d: %s' %(i+1,nX,X_headers[i]) )
    print('\nTarget: ', Y_header)
  
    ########################
    # Prepare other settings
    ########################
    # Setting cross-validation type (either leave-one-group-out or 5-fold)
    if(cv_type=='logo'):
        from sklearn.model_selection import LeaveOneGroupOut
        logo = LeaveOneGroupOut()
        ngroup = logo.get_n_splits(groups=groups)
        print('\nUsing Leave-One-Group-Out cross validation on ', ngroup, ' groups')
    elif(cv_type=='kfold'):
        from sklearn.model_selection import StratifiedKFold
        print('\nUsing 10-fold cross validation')
        k_fold = StratifiedKFold(n_splits=10, random_state=42,shuffle=True)
        cv = k_fold.split(X_data,Y_data)

    #########################
    # Training the regressor
    #########################
    if(gridsearch==True):
        # Finding optimal hyperparameters with GridSearchCV
        if mondrian:
            print('\n Performing GridSearchCV to find optimal hyperparameters for mondrian forest regressor')
            regr = MondrianForestRegressor(**params,random_state=42,bootstrap=False)
            if search_std: # MESSY HACK! Ignore "best model etc" if using this
                def my_scorer(model, X, y_true):
                    y_pred, y_sd = model.predict(X,return_std=True)
                    return np.mean(y_sd)
                scoring=my_scorer
        else:            
            print('\n Performing GridSearchCV to find optimal hyperparameters for random forest regressor')
            regr = RandomForestRegressor(**params,random_state=42)
        if (cv_type=='logo'): cv = logo.split(X_data,Y_data,groups)
        GS_regr = GridSearchCV(estimator=regr,param_grid=GS_params, cv=cv, scoring=scoring, iid=False, verbose=2, **GS_settings)
        GS_regr.fit(X_data,Y_data)

        # Write out results to file
        scores_df = pd.DataFrame(GS_regr.cv_results_)#.sort_values(by='rank_test_score')
        scores_df.to_csv('GridSearch_results.csv')

        # Pich out best results
        best_params = GS_regr.best_params_
        best_score  = GS_regr.best_score_
        regr = GS_regr.best_estimator_  # (this regr has been fit to all of the X_data,Y_data)

        print('\nBest hyperparameters found:', best_params)
        print('\nScore with these hyperparameters:', best_score)

    elif(randomsearch==True):
        # Finding optimal hyperparameters with RandomSearchCV
        if mondrian:
            print('\n Performing RandomizedSearchCV to find optimal hyperparameters for mondrian forest regressor')
            regr = MondrianForestRegressor(**params,random_state=42,bootstrap=False)
        else:            
            print('\n Performing RandomizedSearchCV to find optimal hyperparameters for random forest regressor')
            regr = RandomForestRegressor(**params,random_state=42)
        if (cv_type=='logo'): cv = logo.split(X_data,Y_data,groups)
        RS_regr = RandomizedSearchCV(estimator=regr,param_distributions=RS_params, cv=cv, scoring=scoring,iid=False, verbose=2, error_score=np.nan, **RS_settings)
        RS_regr.fit(X_data,Y_data)
        
        # Write out results to file
        scores_df = pd.DataFrame(RS_regr.cv_results_)#.sort_values(by='rank_test_score')
        scores_df.to_csv('RandomSearch_results.csv')

        # Pick out best results
        best_params = RS_regr.best_params_
        best_score  = RS_regr.best_score_
        regr = RS_regr.best_estimator_  # (this regr has been fit to all of the X_data,Y_data)

        print('\nBest hyperparameters found:', best_params)
        print('\nScore with these hyperparameters:', best_score)


    else:
        # Train RF regressor with hyperparameters given by user
        if mondrian:
            print('\nTraining mondrian forest regressor with given hyperparameters')
            regr = MondrianForestRegressor(**params,bootstrap=False)
        else:            
            print('\nTraining random forest regressor with given hyperparameters')
            regr = RandomForestRegressor(**params)

        # Feature selection before final fit
        if (feature_selection):
            if (cv_type=='logo'): cv = logo.split(X_data,Y_data,groups)
#            [nfeats,scores,traintimes,predtimes], bestscore, bestfeat, featsets = RFE_perm(regr,X_data,Y_data,cv=cv,scoring=scoring,step=step,min_features=min_features,timing=True)
            [nfeats,scores,traintimes,predtimes], bestscore, bestfeat, featsets = RFE_perm(regr,X_data,Y_data,feats,cv=cv,scoring=scoring,timing=True)

            if (scoring=='neg_mean_absolute_error'):
                scores = -scores
                bestscore  = -bestscore
            elif(scoring=='neg_mean_squared_error'):
                scores = np.sqrt(-scores)
                bestscore  = np.sqrt(-bestscore)
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(nfeats,100*scores,lw=2)
            plt.xlabel('$N_{features}$')
            plt.ylabel('Score (%)')
            plt.figure()
            plt.plot(nfeats,traintimes,label='Training',lw=2)
            plt.plot(nfeats, predtimes,label='Prediction',lw=2)
            plt.xlabel('$N_{features}$')
            plt.ylabel('Time (s)')
            plt.legend()
            plt.show()

            print('Best score: %.2f' %(100*bestscore))
            print('Feature set:')
            print(X_headers[bestfeat])

            # Save results in CSV file
            featselect_df = pd.DataFrame(featsets,columns=X_headers)
            featselect_df['score'] = scores
            featselect_df['traintimes'] = traintimes
            featselect_df['predtimes'] = predtimes
            featselect_df['nfeats'] = nfeats
            featselect_df.to_csv('FeatSelect_results.csv')

            # cut down to optimial feature set
            X_data = X_data.iloc[:,bestfeat]

        # Fit model to data
        regr.fit(X_data,Y_data)

    # Cross validation accuracy metrics
    if(accuracy==True):
        print('\nPerforming cross validation to determine train and test accuracy/error')

        # Get generator object depending on cv strategy
        if (cv_type=='logo'): 
            cv = logo.split(X_data,Y_data,groups)
        elif(cv_type=='kfold'):
            cv = k_fold.split(X_data,Y_data)  # Need to regen "Generator" object

        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

        # Init lists
        train_r2  = []
        test_r2   = []
        train_MAE = []
        test_MAE  = []
        train_MSE = []
        test_MSE  = []

        # Loop through CV folds
        i = 0
        for train_index, test_index in cv:
            X_train, X_test = X_data.iloc[train_index], X_data.iloc[test_index]
            Y_train, Y_test = Y_data.iloc[train_index], Y_data.iloc[test_index]

            # Train regressor
            regr_cv = regr
            regr_cv.fit(X_train, Y_train)

            # Predict Y
            Y_pred_train = regr_cv.predict(X_train)
            Y_pred_test  = regr_cv.predict(X_test )

            # r2 scores
            r2score = r2_score(Y_test , Y_pred_test)
            train_r2.append(r2_score(Y_train, Y_pred_train) )
            test_r2.append(r2score)
            # Mean absolute error scores
            MAEscore = mean_absolute_error(Y_test , Y_pred_test)
            train_MAE.append(mean_absolute_error(Y_train, Y_pred_train) )
            test_MAE.append(MAEscore)
            # Mean squared error scores
            MSEscore = mean_squared_error(Y_test , Y_pred_test)
            train_MSE.append(mean_squared_error(Y_train, Y_pred_train) )
            test_MSE.append(MSEscore)

            # Print validation scores (training scores are stored to print mean later, but not printed for each fold)
            if(cv_type=='logo'):
                print('\nTest group = ', groups.iloc[test_index[0]])
            elif(cv_type=='kfold'):
                print('\nFold = ', i)
            print('-------------------')
            print('r2 score = %.2f %%' %(r2score*100) )
            print('Mean absolute error = %.2f %%' %(MAEscore*100) )
            print('Mean squared error = %.2f %%' %(MSEscore*100) )

            i += 1

        # Print performance scores
        print('\nMean training scores:')
        print('r2 score = %.2f %%' %(np.mean(train_r2)*100) )
        print('Mean absolute error = %.2f %%' %(np.mean(train_MAE)*100) )
        print('Mean squared error = %.2f %%' %(np.mean(train_MSE)*100) )
    
        print('\nMean validation scores:')
        print('r2 score = %.2f %%' %(np.mean(test_r2)*100) )
        print('Mean absolute error = %.2f %%' %(np.mean(test_MAE)*100) )
        print('Mean squared error = %.2f %%' %(np.mean(test_MSE)*100) )
        

    return regr
