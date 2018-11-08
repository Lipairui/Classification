# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import re
import jieba
import time
import codecs
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
import lightgbm as lgb
import operator
import matplotlib.pyplot as plt
% matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,recall_score,roc_auc_score,classification_report,average_precision_score,f1_score
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation, metrics
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.linear_model import RidgeClassifier

#cross validation
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score

def LogInfo(stri):
    '''
     Funciton: 
         print log information
     Input:
         stri: string
     Output: 
         print time+string
     '''
    
    print(str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))+'  '+stri)

def RandomForest(X_train, X_test, y_train, y_test):
    X_train = X_train.fillna(-1)
    X_test = X_test.fillna(-1)
    LogInfo('Train RandomForest model...')
#     rf = RandomForestClassifier(min_samples_split=180,n_estimators=130,
#                                       min_samples_leaf=50,max_depth=9,max_features='sqrt',random_state=10)
    rf = RandomForestClassifier(random_state=10)
    rf.fit(X_train,y_train)
    y_prob = rf.predict_proba(X_test)[:,1]
    y_d = [round(i) for i in y_prob]
    print(classification_report(y_test,y_d))
    # Model tunning
    # Grid search
    # n_estimators:  best recall:40  best precision:80
#     param_test1 = {'n_estimators':list(range(10,201,10))}
#     gsearch1 = GridSearchCV(estimator = RandomForestClassifier(min_samples_split=100,
#                                   min_samples_leaf=20,max_depth=8,max_features='sqrt' ,random_state=10), 
#                        param_grid = param_test1, scoring='precision',cv=5)
#     gsearch1.fit(X_train,y_train)
#     print(gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_)

#     Tunning on test dataset
#     for p in param_test1['n_estimators']:
#         print('n_estimators: %d'%p)
#         rf = RandomForestClassifier(min_samples_split=100,n_estimators=p,
#                                       min_samples_leaf=20,max_depth=8,max_features='sqrt' ,random_state=10)
#         rf.fit(X_train,y_train)
#         y_prob = rf.predict_proba(X_test)[:,1]
#         y_d = [round(i) for i in y_prob]
#         print(classification_report(y_test,y_d))
    # best n_estimators: 130
    # best min_samples_split: 180/190
    # best max_depth: 5/7/9/11/13
#     param_test2 = {'max_depth':list(range(3,14,2)), 'min_samples_split':list(range(10,201,10))}
#     for p1 in param_test2['max_depth']:
#         for p2 in param_test2['min_samples_split']:
#             print('max_depth: %d'%p1)
#             print('min_samples_split: %d'%p2)
#             rf = RandomForestClassifier(min_samples_split=p2,n_estimators=130,
#                                           min_samples_leaf=20,max_depth=p1,max_features='sqrt' ,random_state=10)
#             rf.fit(X_train,y_train)
#             y_prob = rf.predict_proba(X_test)[:,1]
#             y_d = [round(i) for i in y_prob]
#             print(classification_report(y_test,y_d))
    
#     param_test3 = {'min_samples_split':range(180,190,5), 'min_samples_leaf':range(10,70,10)}
#     for p1 in param_test3['min_samples_split']:
#         for p2 in param_test3['min_samples_leaf']:
#             print('max_depth: %d'%p1)
#             print('min_samples_split: %d'%p2)
#             rf = RandomForestClassifier(min_samples_split=p1,n_estimators=130,
#                                           min_samples_leaf=p2,max_depth=9,max_features='sqrt' ,random_state=10)
#             rf.fit(X_train,y_train)
#             y_prob = rf.predict_proba(X_test)[:,1]
#             y_d = [round(i) for i in y_prob]
#             print(classification_report(y_test,y_d))
    # best min_samples_leaf:50
#     param_test4 = {'max_features':range(3,20,1)}
#     for p in param_test4['max_features']:
 
#         print('max_features: %d'%p)
#         rf = RandomForestClassifier(min_samples_split=180,n_estimators=130,
#                                       min_samples_leaf=50,max_depth=9,max_features=p ,random_state=10)
#         rf.fit(X_train,y_train)
#         y_prob = rf.predict_proba(X_test)[:,1]
#         y_d = [round(i) for i in y_prob]
#         print(classification_report(y_test,y_d))
    # best max_features:5/'sqrt'
    return y_prob
    
def lightgbm(X_train, X_test, y_train, y_test):
    LogInfo('Train lightgbm model...')
    X_train = X_train.fillna(-1)
    X_test = X_test.fillna(-1)
    params = {'boosting_type': 'gbdt',
           'n_estimators':150,   
          'max_depth' : -1,
          'objective': 'binary',
          'nthread': 3, # Updated from nthread
          'num_leaves': 64,
          'learning_rate': 0.05,
          'max_bin': 512,
          'subsample_for_bin': 200,
          'subsample': 1,
          'subsample_freq': 1,
          'colsample_bytree': 0.8,
          'reg_alpha': 5,
          'reg_lambda': 10,
          'min_split_gain': 0.5,
          'min_child_weight': 1,
          'min_child_samples': 5,
          'scale_pos_weight': 1,
          'num_class' : 1,
          'metric' : 'binary_error'}
    gridParams = {
    'max_depth': list(range(3,12,1)),
#     'learning_rate': [0.005,0.01,0.1],
#     'n_estimators': list(range(50,1000,50)),
    'num_leaves': [6,8,12,16],
    
    
#     'colsample_bytree' : [0.65, 0.7,0.75,0.8],
#     'subsample' : [0.65, 0.7,0.75,0.8],
#     'reg_alpha' : [0.1,0.5,1,1.2],
#     'reg_lambda' : [0.1,0.5,1,1.2,1.4],
    }

    mdl = lgb.LGBMClassifier(boosting_type= 'gbdt',
          objective = 'binary',
          n_jobs = 3, # Updated from 'nthread'
          silent = True,
          random_state = 10,                   
          metric = params['metric'],                   
          n_estimators = params['n_estimators'],
          learning_rate = params['learning_rate'],
          max_depth = params['max_depth'],
          max_bin = params['max_bin'],
          subsample_for_bin = params['subsample_for_bin'],
          subsample = params['subsample'],
          subsample_freq = params['subsample_freq'],
          min_split_gain = params['min_split_gain'],
          min_child_weight = params['min_child_weight'],
          min_child_samples = params['min_child_samples'],
          scale_pos_weight = params['scale_pos_weight'])
    grid = GridSearchCV(mdl, gridParams,
                    verbose=0,
                    cv=4,
                    n_jobs=2)
    
    # Run the grid
    grid.fit(X_train,y_train)
    # Print the best parameters found
    print(grid.best_params_)
    print(grid.best_score_)
    
    # Using parameters already set above, replace in the best from the grid search
#     params['n_estimators'] = grid.best_params_['n_estimators']
#     params['colsample_bytree'] = grid.best_params_['colsample_bytree']
#     params['learning_rate'] = grid.best_params_['learning_rate']
    # params['max_bin'] = grid.best_params_['max_bin']
    params['num_leaves'] = grid.best_params_['num_leaves']
    
#     params['reg_alpha'] = grid.best_params_['reg_alpha']
#     params['reg_lambda'] = grid.best_params_['reg_lambda']
#     params['subsample'] = grid.best_params_['subsample']
    mdl = lgb.LGBMClassifier(boosting_type= 'gbdt',
          objective = 'binary',
          n_jobs = 3, # Updated from 'nthread'
          silent = True,
          random_state = 10,                   
          metric = params['metric'],                   
          n_estimators = params['n_estimators'],
          learning_rate = params['learning_rate'],
          max_depth = params['max_depth'],
          max_bin = params['max_bin'],
          subsample_for_bin = params['subsample_for_bin'],
          subsample = params['subsample'],
          subsample_freq = params['subsample_freq'],
          min_split_gain = params['min_split_gain'],
          min_child_weight = params['min_child_weight'],
          min_child_samples = params['min_child_samples'],
          scale_pos_weight = params['scale_pos_weight'])
    mdl.fit(X_train,y_train)
    y_prob = mdl.predict_proba(X_test)[:,1]
    y_d = [round(i) for i in y_prob]
    print(classification_report(y_test,y_d))
    
    
def GBDT(X_train, X_test, y_train, y_test):
    X_train = X_train.fillna(-1)
    X_test = X_test.fillna(-1)
    LogInfo('Train GBDT model...')
    gbm = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=320,n_estimators=1250,
                              min_samples_leaf=45,max_depth=3,max_features='sqrt', subsample=0.8,random_state=10)
    gbm.fit(X_train,y_train)
    y_prob = gbm.predict_proba(X_test)[:,1]
    y_d = [round(i) for i in y_prob]
    print(classification_report(y_test,y_d))
#     Model tunning on test dataset
#     for n in range(100,2000,50):
#         print('n_estimators: %d'%n)
#         gbm = GradientBoostingClassifier(learning_rate=0.01, n_estimators=n, min_samples_leaf=15, 
#                                     max_depth=8,min_samples_split=240,max_features='sqrt', subsample=0.8, random_state=10)
#         gbm.fit(X_train,y_train)
#         y_prob = gbm.predict_proba(X_test)[:,1]
#         y_d = [round(i) for i in y_prob]
#         print(classification_report(y_test,y_d))
#     param_test1 = {'n_estimators':range(5,200,5)}
#     for p in param_test1['n_estimators']:
#         print('n_estimators: %d'%p)
#         gbm = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=300,n_estimators=p,
#                                   min_samples_leaf=20,max_depth=8,max_features='sqrt', subsample=0.8,random_state=10)
#         gbm.fit(X_train,y_train)
#         y_prob = gbm.predict_proba(X_test)[:,1]
#         y_d = [round(i) for i in y_prob]
#         print(classification_report(y_test,y_d))
    # best n_estimators: 1250
#     param_test2 = {'max_depth':range(3,5,1), 'min_samples_split':range(200,350,50)}
#     for p1 in param_test2['max_depth']:
#         for p2 in param_test2['min_samples_split']:
#             print('max_depth: %d'%p1)
#             print('min_samples_split: %d'%p2)
#             gbm = GradientBoostingClassifier(learning_rate=0.1, n_estimators=1250, min_samples_leaf=20, 
#                                              max_depth=p1,min_samples_split=p2,max_features='sqrt', subsample=0.8, random_state=10)
#             gbm.fit(X_train,y_train)
#             y_prob = gbm.predict_proba(X_test)[:,1]
#             y_d = [round(i) for i in y_prob]
#             print(classification_report(y_test,y_d))
    # best max_depth:3 best min_samples_split:200
#     param_test3 = {'min_samples_split':range(200,400,40), 'min_samples_leaf':range(10,120,5)}
#     for p1 in param_test3['min_samples_split']:
#         for p2 in param_test3['min_samples_leaf']:
#             print('min_samples_split: %d'%p1)
#             print('min_samples_leaf: %d'%p2)
#             gbm = GradientBoostingClassifier(learning_rate=0.1, n_estimators=1250, min_samples_leaf=p2, 
#                                              max_depth=3,min_samples_split=p1,max_features='sqrt', subsample=0.8, random_state=10)
#             gbm.fit(X_train,y_train)
#             y_prob = gbm.predict_proba(X_test)[:,1]
#             y_d = [round(i) for i in y_prob]
#             print(classification_report(y_test,y_d))
    # best min_samples_split:320 min_samples_leaf:45
#     param_test4 = {'max_features':range(7,20,2)}
#     for p in param_test4['max_features']:
#         print('max_features: %d'%p)
#         gbm = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=320,n_estimators=1250,
#                               min_samples_leaf=45,max_depth=3,max_features=p, subsample=0.8,random_state=10)
#         gbm.fit(X_train,y_train)
#         y_prob = gbm.predict_proba(X_test)[:,1]
#         y_d = [round(i) for i in y_prob]
#         print(classification_report(y_test,y_d))
    return y_prob


def XGBoost(X_train, X_test, y_train, y_test):
    LogInfo('Train xgboost model...')
#     dtrain = xgb.DMatrix(X_train.drop(['sex_words'],axis=1),y_train)
#     dtest = xgb.DMatrix(X_test.drop(['sex_words'],axis=1))
    X_train = X_train.astype(float)
    X_test = X_test.astype(float)
    dtrain = xgb.DMatrix(X_train,y_train)
    dtest = xgb.DMatrix(X_test)
    # specify parameters
#     params = {'max_depth':6,'eta':1,'silent':0,'objective':'binary:logistic'}
    params = {
        'objective':'binary:logistic',
        'min_child_weight': 5,
        'learning_rate': 0.1,
        'max_depth': 5,
        'subsample': 0.9,      
        'colsample_bytree':0.9,
        'gamma': 0.4,
        'reg_alpha':0.01,
        'scale_pos_weight':1,
        'verbose_eval': True,
        'seed': 12,
        'silent':1
    }

    num_rounds = 520
    
#     param_test1 = {
#      'max_depth':[5],
#      'reg_alpha':[0, 0.001, 0.005, 0.01,0.03, 0.05]# 0.01
#     }
#     p1 = list(param_test1.keys())[0]
#     p2 = list(param_test1.keys())[1]
#     for v1 in param_test1[p1]:
#         for v2 in param_test1[p2]:
#             print(p1+': %f'%v1)
#             print(p2+': %f'%v2)
#             params[p1] = v1
#             params[p2] = v2      
#             # Train
#             bst = xgb.train(params, dtrain, num_rounds)
#             # predict
#             pred = bst.predict(dtest)
#             y_d = [round(p) for p in pred]
#             # evaluate
#             LogInfo('Evaluate...')
#             print(classification_report(y_test,y_d))
    # Train
    bst = xgb.train(params, dtrain, num_rounds)
    # predict
    pred = bst.predict(dtest)
    y_d = [round(p) for p in pred]
    # evaluate
    LogInfo('Evaluate...')
    print(classification_report(y_test,y_d))

#     # show feature importance
#     features = X_train.columns.values
#     LogInfo('Feature dims: '+str(len(features)))
#     plot_feature_importance(bst,features)
    return pred
    
    
# Plot XGBoost feature importance
def ceate_feature_map(features):
    outfile = open('../res/xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1
    outfile.close()

def plot_feature_importance(bst,features):
    '''
    Input:
        bst: trained XGBoost model
        features: list of feature names
    '''
    ceate_feature_map(features)
 
    importance = bst.get_fscore(fmap='../res/xgb.fmap')
    importance = sorted(importance.items(), key=operator.itemgetter(1))
 
    df = pd.DataFrame(importance, columns=['feature', 'fscore'])
    df['fscore'] = df['fscore'] / df['fscore'].sum()
    df.to_csv("../res/feature_importance3.csv",index=False)
    plt.figure()
    df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
    plt.title('XGBoost Feature Importance')
    plt.xlabel('relative importance')
    plt.show()
    plt.savefig('../res/feature_importance3.jpg')


# Stacking model
def generate_clf_features(clf, clf_name, X_train, X_test, y_train):
    LogInfo('Generate '+clf_name+' features...')
    random_seed = 2016     
    skf = StratifiedKFold(y_train, n_folds=5, shuffle=True)

    new_train = np.zeros((X_train.shape[0],1))
    new_test = np.zeros((X_test.shape[0],1))

    for i,(trainid,valid) in enumerate(skf):
        print('fold ' + str(i))
        train_x = X_train.iloc[trainid]
        train_y = y_train[trainid]
        val_x = X_train.iloc[valid]
        clf.fit(train_x, train_y)
        if clf_name== 'svc':
            new_train[valid] = clf.decision_function(val_x).reshape(-1,1)
            new_test += clf.decision_function(X_test).reshape(-1,1)
        else:
            new_train[valid] = clf.predict_proba(val_x)[:,0].reshape(-1,1)
            new_test += clf.predict_proba(X_test)[:,0].reshape(-1,1)

    new_test /= 5
    stacks = []
    stacks_name = []
    print(len(new_train),len(new_test))
    stack = np.vstack([new_train,new_test])
    stacks.append(stack)
    stacks_name = [clf_name]
    stacks = np.hstack(stacks)
    clf_stacks = pd.DataFrame(data=stacks,columns=stacks_name)   
    path = '../features/'+clf_name+'_prob1.csv'
    clf_stacks.to_csv(path, index=0)
    return clf_stacks

def stacking(X_train, X_test, y_train, y_test):
    #Naive Bayes models
    mnb_clf = MultinomialNB()
    gnb_clf = GaussianNB()
    bnb_clf = BernoulliNB()

    #SVM_based models
    sgd_clf = SGDClassifier(loss = 'hinge',penalty = 'l2', alpha = 0.0001,n_iter = 500, random_state = 42, verbose=1, n_jobs=256)

    svc_clf = svm.LinearSVC(penalty='l2', loss='hinge', dual=True, tol=0.000001, C=0.5, multi_class='ovr', 
                             fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=1, random_state=None, max_iter=5000)

    svm_clf = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=True, 
                      tol=0.001, cache_size=200, class_weight=None, verbose=True, max_iter=-1, decision_function_shape='ovr', random_state=None)

    #Logistic Regression
    lr_clf = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, 
                                class_weight=None, random_state=None, solver='liblinear', max_iter=5000, 
                                multi_class='ovr', verbose=1, warm_start=False, n_jobs=256)

    X_train = X_train.fillna(-1)
    X_test = X_test.fillna(-1)
    model = {'lr':lr_clf,'bnb':bnb_clf}
    for m in model.keys():
        # Get features
        features = generate_clf_features(model[m],m,X_train,X_test, y_train)
        # Evaluate
        LogInfo('Evaluate '+m)
        y_d = [1 if y > 0.5 else 0 for y in features.iloc[len(X_train):].values]
        print(classification_report(y_test,y_d))
    
