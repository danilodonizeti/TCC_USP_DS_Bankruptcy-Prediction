# -*- coding: utf-8 -*-

''' Importa as bibliotecas '''
import pandas as pd
import numpy as np

from sklearn.model_selection import ParameterGrid
from IPython.display import clear_output, Markdown, display
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

from IPython.display import clear_output

from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

''' Colunas do relatório com os resultados '''
columns = ['Nome', 'Alg-Class', 'Alg-Class-Params', 'Tec-Desbal', 'Tec-Desbal-Params', 'Acuracia', 'Acuracia Balanceada', 
           'F1-Score', 'Macro F1-Score', 'AUC']

''' Calcula o número de classificadores do arquivo com a configuração dos classificadores '''
def qtde_classificadores(classifier_settings):
    ''' Retorna a quantidade de classificadores dado o arquivo "settings.py"
    '''
    n_classifiers = 0
    for _, settings in classifier_settings.items():
        for _, (_, param_grid) in settings.items():
            grid = ParameterGrid(param_grid)
            for _ in grid:
                n_classifiers += 1
    return n_classifiers

''' Classe que executa o experimento '''
class ExecutaExperimento():
    ''' Esse método inicializa a classe 
    '''
    def __init__(self, clf_config, n=5, nome_experimento='exp', random_state=42):
        self.clf_config = clf_config
        self.n = n
        self.nome_experimento = nome_experimento
        self.random_state = random_state
        self.kf = StratifiedKFold(n_splits=n)
        self.relatorio = None
        self.contador = 0
        p = 9
        self.n_classificadores = qtde_classificadores(self.clf_config())*p # esse valor p é quantas vezes o balanceamento é rodado por classificador  

    def display_progresso(self, clf_name, i):
        clear_output()
        print('Fold %d/%d | Classificador %d/%d (%s)' %  (i, self.n, self.contador, self.n_classificadores, clf_name))

    ''' Esse método basicamente roda o experimento, primeiro prepara o relatório, executa o kfold e depois salva o relatório
    '''
    def executa(self, X, y):
        self.inicializa_relatorio()
        self.kfold(X, y)
        self.salva_relatorio()
    
    def inicializa_relatorio(self):
        self.relatorio = pd.DataFrame(columns=columns)
        
    def atualiza_relatorio(self, clf_tipo, clf_params, bal_tipo, bal_params, valores):
        nome_clf = 'CLF_' + str(self.contador)
        aux = pd.DataFrame([[nome_clf, clf_tipo, str(clf_params), bal_tipo, str(bal_params)] + valores], columns=columns)
        self.relatorio = pd.concat([self.relatorio, aux], ignore_index=True)
        self.contador += 1 # atualiza o contador com o nome do classificador
        
    def salva_relatorio(self):
        self.relatorio = self.relatorio.groupby(by=['Nome', 'Alg-Class', 'Alg-Class-Params', 'Tec-Desbal', 'Tec-Desbal-Params']).mean()
        self.relatorio = self.relatorio.reset_index()
        self.relatorio.to_csv(self.nome_experimento + '.csv', sep=';', index=False)

    ''' Calcula o conjunto de métricas pré-definidos para o relatório '''
    def calcula_metricas(self, x_test, y_test, clf):
        y_predict = clf.predict(x_test)
        acuracia = accuracy_score(y_test, y_predict)
        acuracia_b = balanced_accuracy_score(y_test, y_predict)
        aux_f1_score = f1_score(y_test, y_predict)
        macro_f1 = f1_score(y_test, y_predict, average='macro')
        auc = roc_auc_score(y_test, clf.predict_proba(x_test)[:, 1])
        return [acuracia, acuracia_b, aux_f1_score, macro_f1, auc]

    ''' Executa a classificação do classificador em questão, sem nenhum método de balanceamento '''
    def classifica_sem_balanceamento(self, x_train, x_test, y_train, y_test, clf, clf_tipo, clf_params):
        clf.fit(x_train, y_train)
        valores = self.calcula_metricas(x_test, y_test, clf)
        self.atualiza_relatorio(clf_tipo, clf_params, '-', '-', valores)

    def classifica_com_randomundersample(self, x_train, x_test, y_train, y_test, clf, clf_tipo, clf_params):
        bal = RandomUnderSampler(random_state=self.random_state)
        x_train_resampled, y_train_resampled = bal.fit_resample(x_train, y_train)
        clf.fit(x_train_resampled, y_train_resampled)
        valores = self.calcula_metricas(x_test, y_test, clf)
        self.atualiza_relatorio(clf_tipo, clf_params, 'undersampling', '-', valores)

    def classifica_com_randomoversample(self, x_train, x_test, y_train, y_test, clf, clf_tipo, clf_params):
        bal = RandomOverSampler(random_state=self.random_state)
        x_train_resampled, y_train_resampled = bal.fit_resample(x_train, y_train)
        clf.fit(x_train_resampled, y_train_resampled)
        valores = self.calcula_metricas(x_test, y_test, clf)
        self.atualiza_relatorio(clf_tipo, clf_params, 'oversampling', '-', valores)

    def classifica_com_smote(self, x_train, x_test, y_train, y_test, clf, clf_tipo, clf_params):
        # coloca variações nas configurações do smote
        for value in [5, 10, 15]:
            bal = SMOTE(random_state=self.random_state, k_neighbors=value)
            x_train_resampled, y_train_resampled = bal.fit_resample(x_train, y_train)
            clf.fit(x_train_resampled, y_train_resampled)
            valores = self.calcula_metricas(x_test, y_test, clf)
            self.atualiza_relatorio(clf_tipo, clf_params, 'smote', value, valores)

    def classifica_com_adasyn(self, x_train, x_test, y_train, y_test, clf, clf_tipo, clf_params):
        # coloca variações nas configurações do adasyn
        for value in [5, 10, 15]:
            bal = ADASYN(random_state=self.random_state, n_neighbors=value)
            x_train_resampled, y_train_resampled = bal.fit_resample(x_train, y_train)
            clf.fit(x_train_resampled, y_train_resampled)
            valores = self.calcula_metricas(x_test, y_test, clf)
            self.atualiza_relatorio(clf_tipo, clf_params, 'adasyn', value, valores)

    ''' Código com a validação cruzada, para cada cada fold, algoritmo de classificação e configuração roda todas as 
        técnicas de balanceamento selecionadas e também roda sem o balanceamento para utilizar como baseline de comparação.
    '''  
    def kfold(self, X, y):
        ''' Roda todos os folds '''
        for i, (train_index, test_index) in enumerate(self.kf.split(X, y)):
            self.contador = 0
            x_train, x_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]
            ''' Para cada fold, roda todos os modelos '''
            all_configs = self.clf_config()
            for config_name, settings in all_configs.items():
                for clf_tipo, (Classifier, param_grid) in settings.items():
                    grid = ParameterGrid(param_grid)
                    for clf_params in grid:
                        ''' Roda o modelo em questão '''
                        self.display_progresso(clf_tipo, i)
                        clf =  Classifier(**clf_params)
                        self.classifica_sem_balanceamento(x_train, x_test, y_train, y_test, clf, clf_tipo, clf_params)
                        self.classifica_com_randomundersample(x_train, x_test, y_train, y_test, clf, clf_tipo, clf_params)
                        self.classifica_com_randomoversample(x_train, x_test, y_train, y_test, clf, clf_tipo, clf_params)
                        self.classifica_com_smote(x_train, x_test, y_train, y_test, clf, clf_tipo, clf_params)
                        self.classifica_com_adasyn(x_train, x_test, y_train, y_test, clf, clf_tipo, clf_params)
                        # se for incluir mais técnicas de balanceamento colocar aqui a chamada do método e fazer como os métodos acima
                                  