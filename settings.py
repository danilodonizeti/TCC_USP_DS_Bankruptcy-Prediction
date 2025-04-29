import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier


def set_configs():

    arvore_decisao = {
        'AD' : [DecisionTreeClassifier,
                {'min_samples_leaf' : list(range(1, 20, 2)), 'random_state' : [42]}]
    }

    knn = {
        'KNN' : [KNeighborsClassifier,
                 {'n_neighbors' : [1, 3, 5, 7, 11, 13, 17, 19, 23], 'n_jobs' : [-1]}]
        
    }

    mlp = {
        'MLP' : [MLPClassifier,
                 {'hidden_layer_sizes' : list(range(5, 24, 2)), 'random_state' : [42]}]
    }
    
    random_forest = {
        'RF' : [RandomForestClassifier,
                {'n_estimators' : list(range(50, 550, 50)), 'random_state' : [42], 'n_jobs' : [-1]}]
    }

    regressao_logistica = {
        'RL' : [LogisticRegression,
               {'C' : list(np.arange(0.8, 1.30, 0.05)), 'random_state' : [42], 'n_jobs' : [-1]}]
    
    }
    
    all_configs = {
        'ad'  : arvore_decisao,
        'knn' : knn,
        'mlp' : mlp,
        'rf'  : random_forest,
        'rl'  : regressao_logistica
    }
    
    return all_configs