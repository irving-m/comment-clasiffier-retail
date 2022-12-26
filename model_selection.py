# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 21:26:55 2022

@author: Irving
"""


import os
import pandas as pd
import numpy as np
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier

from spell_fix import lookup


path = os.getcwd()

with open(os.path.join(path, "comments_set.xlsx"), "rb") as g:
    data = pd.read_excel(g)
    

use = data[["Input", "Label"]]
use["Input"] = use["Input"].str.replace(r"[^\w\s]", "")
use["Input"] = use["Input"].replace("", np.nan)

df = use.dropna()
df["fixed"] = df["Input"].apply(lookup)

labels = list(set(df["Label"]))

def accuracy_multiclass(labels, y_true, x_test, model):
    mat = confusion_matrix(y_true, model.predict(x_test),
                           labels= labels)
    values = {}
    for i, label in enumerate(labels):
        values[label] = mat[i, i]/ sum(mat[i,:])
        
    return values


tests = pd.DataFrame(labels, columns= ["Clases"])
    
# PARTICIÓN DE DATOS
randomstate = 253
X_train, X_test, y_train, y_test = train_test_split(df["fixed"], df["Label"],
                                                    stratify= df["Label"],
                                                    random_state= randomstate)

stop = ["el", "la", "las", "los", "y", "o", "le", "me", "un", "una", "en",
        "su", "sus", "del", "mas", "fue", "d", "por", "para", "mi", "con",
        "no", "al", "se", "nos", "es", "para", "que", "son", "si", "sí"]

vectorizer = TfidfVectorizer(decode_error= "ignore",
                             strip_accents= "ascii",
                             lowercase= True,
                             min_df= 5,
                             stop_words= stop)


# DECISION TREE CLASSIFIER
tree_clf = Pipeline([("vec", vectorizer),
                     ("tree", DecisionTreeClassifier(random_state= 14))])

tree_grid = {"tree__min_samples_split": [30, 35, 40],
              "tree__criterion": ["gini", "entropy", "log_loss"],
              "tree__class_weight": ["balanced", None]}

tree = GridSearchCV(tree_clf,
                    param_grid= tree_grid,
                    scoring= "precision_weighted",
                    cv= 5,
                    verbose= 2,
                    n_jobs= -1)

tree.fit(X_train, y_train)
tree.best_params_
tests["tree"] = accuracy_multiclass(labels, y_test, X_test,
                                    tree).values()

# LOGISTIC CLASSIFICATION
log_clf = Pipeline([("vec", vectorizer),
                     ("log", LogisticRegression(random_state= 14, 
                                                max_iter= 500))])

log_grid = {"log__C": [3, 4, 5],
            "log__solver": ["newton-cg", "lbfgs", "saga"],
            "log__penalty": ["l2", None],
            "log__multi_class": ["ovr", "multinomial"],
            "log__class_weight": ["balanced", None]}

log = GridSearchCV(log_clf,
                   param_grid= log_grid,
                   scoring= "precision_weighted",
                   cv= 5,
                   verbose= 2,
                   n_jobs= -1)

log.fit(X_train, y_train)
log.best_params_
tests["log"] = accuracy_multiclass(labels, y_test, X_test,
                                    log).values()


# RANDOM FOREST CLASSIFIER
forest_clf = Pipeline([("vec", vectorizer),
                       ("forest", RandomForestClassifier(random_state= 14))])

forest_grid = {"forest__n_estimators": [150, 175, 200],
               "forest__min_samples_split": [10, 15, 20],
               "forest__class_weight": [None, "balanced", "balanced_subsample"],
               "forest__criterion": ["entropy", "gini"],
               "forest__bootstrap": [True, False]}

forest = GridSearchCV(forest_clf,
                      param_grid= forest_grid,
                      scoring= "precision_weighted",
                      cv= 5,
                      verbose= 2,
                      n_jobs= -1)

forest.fit(X_train, y_train)
forest.best_params_
tests["forest"] = accuracy_multiclass(labels, y_test, X_test,
                                      forest).values()



# SUPPORT VECTOR CLASSIFICATION
svc_clf = Pipeline([("vec", vectorizer),
                     ("svc", LinearSVC(random_state= 14))])

svc_grid = {"svc__C": [0.5, 0.6, 0.7],
            "svc__penalty": ["l1", "l2"],
            "svc__loss": ["hinge", "squared_hinge"],
            "svc__multi_class": ["ovr", "crammer_singer"],
            "svc__class_weight": ["balanced", None]}

svc = GridSearchCV(svc_clf,
                   param_grid= svc_grid,
                   scoring= "precision_weighted",
                   cv= 5,
                   verbose= 3,
                   n_jobs= -1)

svc.fit(X_train, y_train)
svc.best_params_
tests["svc"] = accuracy_multiclass(labels, y_test, X_test,
                                   svc).values()


# K NEAREST NEIGHBORS
knn_clf = Pipeline([("vec", vectorizer),
                     ("knn", KNeighborsClassifier())])

knn_grid = {"knn__n_neighbors": [1, 2, 3],
            "knn__leaf_size": [1, 2, 3],
            "knn__p": [1, 2, 3],
            "knn__weights": ["uniform", "distance"],
            "knn__algorithm": ["auto"]}
            
knn = GridSearchCV(knn_clf,
                   param_grid= knn_grid,
                   scoring= "precision_weighted",
                   cv= 5,
                   verbose= 2,
                   n_jobs= -1)

knn.fit(X_train, y_train)
knn.best_params_
tests["knn"] = accuracy_multiclass(labels, y_test, X_test,
                                   knn).values()


# GRADIENT BOOST CLASSIFICATION

gb_clf = Pipeline([("vec", vectorizer),
                   ("gb", GradientBoostingClassifier(random_state= 14))])

gb_grid = {"gb__n_estimators": [100, 120, 140, 160],
           "gb__subsample": [1],
           "gb__min_samples_split": [25, 30, 40],
           "gb__loss": ["log_loss", "exponential"]}
            
gb = GridSearchCV(gb_clf,
                  param_grid= gb_grid,
                  scoring= "precision_macro",
                  cv= 5,
                  verbose= 3)

gb.fit(X_train, y_train)
gb.best_params_
tests["gb"] = accuracy_multiclass(labels, y_test, X_test,
                                  gb).values()

   
with open(os.path.join(path, "results.xlsx"), "wb") as f:
    tests.to_excel(f, sheet_name= "Raw")

with open(os.path.join(path, "linearsvc.pickle"), "wb") as f:
    pickle.dump(svc.best_estimator_, f)

 