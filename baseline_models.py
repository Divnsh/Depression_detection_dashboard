from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split,GridSearchCV, StratifiedKFold
from sklearn import metrics
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import xgboost
import os
import random
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression,PassiveAggressiveClassifier
from sklearn.decomposition import TruncatedSVD
import pickle
from scipy.sparse import csr_matrix,hstack,save_npz,load_npz

DATA_DIR='./Data'
data=pd.read_csv(os.path.join(DATA_DIR,"cleaned_reddit.csv"))
data=data.drop_duplicates().dropna()

MODELS_DIR='./Models'
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

## Modelling ##

# Train-val-test split
# Split into 70,15,15 splits

# train_idx = idx[:int(.7*len(data))]
# val_idx = idx[int(.7*len(data)):int(.85*len(data))]
# test_idx = idx[int(.85*len(data)):]

text_to_use='treated_stemmed_text'
#text_to_use='treated_lemmatized_text'
#text_to_use='stemmed_text'
#text_to_use='lemmatized_text'
#text_to_use='cleaned_text'

# mask=data.label==1
# data.loc[mask,text_to_use] = data.loc[mask,text_to_use].apply(lambda x: ' '.join([w for w in x.split() if 'depress' not in w]))

X_train, X_val, y_train, y_val = train_test_split(data[[text_to_use]], data['label'],
                                    test_size=0.30, random_state=2020, stratify=data['label'])

X_val, X_test, y_val, y_test = train_test_split(X_val, y_val,
                                    test_size=0.5, random_state=2020, stratify=y_val)
del data

# Feature engineering
# Tf-idf

def get_tfidf(train,test):
    ngram_limit=3
    tfidf=TfidfVectorizer(min_df=5, max_features=None, strip_accents='unicode',analyzer='word',
                            token_pattern=r'\w{1,}', ngram_range=(1,ngram_limit), use_idf=1, smooth_idf=1,
                            sublinear_tf=1, stop_words='english')

    X_train_tfidf=tfidf.fit_transform(train[text_to_use])
    X_val_tfidf=tfidf.transform(test[text_to_use])
    return X_train_tfidf,X_val_tfidf

X_train_tfidf,X_val_tfidf=get_tfidf(X_train,X_val)
X_train_tfidf,X_test_tfidf=get_tfidf(X_train,X_test)


# LSA

def get_LSA(train,test):
    svd = TruncatedSVD(n_components=120)
    X_train_svd=svd.fit_transform(train)
    X_val_svd=svd.transform(test)
    return csr_matrix(X_train_svd),csr_matrix(X_val_svd)

X_train_svd,X_val_svd=get_LSA(X_train_tfidf,X_val_tfidf)
X_train_svd,X_test_svd=get_LSA(X_train_tfidf,X_test_tfidf)

# Build features

X_train = hstack((X_train_tfidf,X_train_svd))
X_val = hstack((X_val_tfidf,X_val_svd))
X_test = hstack((X_test_tfidf,X_test_svd))

del X_train_tfidf,X_val_tfidf,X_test_tfidf,X_train_svd,X_val_svd,X_test_svd
# save_npz(os.path.join(DATA_DIR,"X_train.npz"), X_train)
# save_npz(os.path.join(DATA_DIR,"X_val.npz"), X_val)
# save_npz(os.path.join(DATA_DIR,"X_test.npz"), X_test)
#
# X_train = load_npz(os.path.join(DATA_DIR,"X_train.npz")).tocsr()
# X_val = load_npz(os.path.join(DATA_DIR,"X_val.npz")).tocsr()
# X_test = load_npz(os.path.join(DATA_DIR,"X_test.npz")).tocsr()

# Model fitting
def print_results(preds, target, labels, sep='-', sep_len=40, fig_size=(10,8)):
    print('Accuracy = %.3f' % metrics.accuracy_score(target, preds))
    print(sep*sep_len)
    print('Classification report:')
    print(metrics.classification_report(target, preds))
    print(sep*sep_len)
    print('Confusion matrix')
    cm=metrics.confusion_matrix(target, preds)
    cm = cm / np.sum(cm, axis=1)[:,None]
    sns.set(rc={'figure.figsize':fig_size})
    sns.heatmap(cm,
        xticklabels=labels,
        yticklabels=labels,
           annot=True, cmap = 'YlGnBu')
    plt.pause(0.05)

def fit_and_predict(clf,params):
    grid_search = GridSearchCV(clf, param_grid=params, scoring='accuracy', n_jobs=-1, cv=3, verbose=3,
                               return_train_score = True)
    if hasattr(clf,'max_depth'):
        grid_search.fit(xgtrain, y_train)
        print("mean_train_score:", grid_search.cv_results_['mean_train_score'])
        print("mean_test_score:", grid_search.cv_results_['mean_test_score'])
        print("best parameters:", grid_search.best_params_)
        clf_val = grid_search.best_estimator_
        preds1 = clf_val.predict(xgval)
        print("Validation results ------------>")
        print_results(y_val, preds1, clf_val.classes_)
        preds2 = clf_val.predict(xgtest)
        print("Test results -------------->")
        print_results(y_test, preds2, clf_val.classes_)
    else:
        grid_search.fit(X_train, y_train)
        print("mean_train_score:", grid_search.cv_results_['mean_train_score'])
        print("mean_test_score:", grid_search.cv_results_['mean_test_score'])
        print("best parameters:", grid_search.best_params_)
        clf_val = grid_search.best_estimator_
        preds1 = clf_val.predict(X_val)
        print("Validation results ------------>")
        print_results(y_val, preds1, clf_val.classes_)
        preds2 = clf_val.predict(X_test)
        print("Test results -------------->")
        print_results(y_test, preds2, clf_val.classes_)
    return clf_val


# SVM # too slow
params = {'kernel':['linear','rbf','poly', 'sigmoid'], 'C':[1, 50,100]}
clf=SVC(degree=3, # degree of polynomial
         gamma=1, # kernel coefficient
         coef0=1, # change to 1 from default value of 0.0
         shrinking=True, # using shrinking heuristics
         tol=0.001, # stopping criterion tolerance
         probability=False, # no need to enable probability estimates
         cache_size=200, # 200 MB cache size
         class_weight=None, # all classes are treated equally
         max_iter=-1, # no limit
         random_state=2020)
clf_val=fit_and_predict(clf,params)
with open(os.path.join(MODELS_DIR,'svm_'+text_to_use+'model'),'wb') as f:
    pickle.dump(clf_val,f)
# Linear kernel - train: test:


# PassiveAggressive
params={'C':[.2,1,10,50]}
clf=PassiveAggressiveClassifier(C=1)
clf_val=fit_and_predict(clf,params)
with open(os.path.join(MODELS_DIR,'passiveaggressive_'+text_to_use+'model'),'wb') as f:
    pickle.dump(clf_val,f)

# Logistic Regression
params ={'penalty' : ['l1', 'l2'],
    'C' : np.logspace(-4, 4, 10),
    'solver' : ['liblinear']}
clf=LogisticRegression()
clf_val=fit_and_predict(clf,params)
with open(os.path.join(MODELS_DIR,'logistic_'+text_to_use+'model'),'wb') as f:
    pickle.dump(clf_val,f)

# XGBoost
params = {
    'max_depth': [3,5,7,10],  # the maximum depth of each tree
    'learning_rate':[0.1],
    'objective': ['binary:logistic'],
    'n_estimators': [1000],
    'min_child_weight' : [1,3,5,7,9],
    'tree_method':['gpu_hist'], # Running on GPU
    'seed': [2020],
    'predictor': ['cpu_predictor']
    }
clf = xgboost.XGBClassifier()
xgtrain = X_train.tocsr()[:,-120:]
xgtest = X_test.tocsr()[:,-120:]
xgval = X_val.tocsr()[:,-120:]
clf_val=fit_and_predict(clf,params)
with open(os.path.join(MODELS_DIR,'xgboost_'+text_to_use+'.model'),'wb') as f:
    pickle.dump(clf_val,f)

print(f"All models saved at: {MODELS_DIR}")


clf = xgboost.XGBClassifier(max_depth=10,learning_rate=.1,objective='binary:logistic',n_estimators=1000,min_child_weight=5,
                            tree_method='gpu_hist',seed=2020,predictor='cpu_predictor')
clf.fit(xgtrain,y_train)
preds=clf.predict(xgval)
print(metrics.classification_report(y_val,preds))
preds=clf.predict(xgtest)
print(metrics.classification_report(y_test,preds))

# XGBoost - val:0.85 test:0.87



