''' Importa as bibliotecas '''
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler                    

class Bankruptcy():
    ''' Classe do conjunto de dados "Bankruptcy" '''
    def __init__(self):
        self.dataset = self._open_dataset()
        self.class_attribute = 'Bankrupt?'
    
    def remove_rows_with_nan(self, symbol=np.nan):
        ''' Remove exemplos com valores nulos '''
        n_rows = self.dataset.shape[0]
        self.dataset = self.dataset.replace(symbol, np.nan)
        self.dataset = self.dataset.dropna()

    def remove_duplicates(self):
        ''' Remove exemplos duplicados '''
        n_rows = self.dataset.shape[0]
        ''' Conserva um dos exemplos duplicados - keep = 'first' '''
        self.dataset = self.dataset.drop_duplicates(self.dataset.columns, keep='first')

    def remove_inconsistent_class(self):
        ''' Remove exemplos com classes inconsistentes '''
        n_rows = self.dataset.shape[0]
        columns = list(self.dataset.columns)
        columns.remove(self.class_attribute)
        self.dataset = self.dataset.drop_duplicates(subset=columns, keep=False)
            
    def get_xy(self):
        x = self.dataset.loc[:, self.dataset.columns != self.class_attribute]
        y = np.array(self.dataset.loc[:, self.dataset.columns == self.class_attribute]).ravel()
        return x, y

    def _open_dataset(self):
        dataset = pd.read_csv('data.csv', index_col=False, sep=',')
        return dataset

    def basic_preprocessing(self):
        ''' Performa os processamentos básicos, importante notar que as três primeiras linhas não exclui nenhum exemplo nesse conjunto de dados 
            Basicamente esse conjunto de dados já está pré-processado, foi colocado algumas coisas aqui, mas sem efeito. 
        '''
        self.remove_rows_with_nan()
        self.remove_duplicates()
        self.remove_inconsistent_class()
        self.dataset = pd.get_dummies(self.dataset, prefix_sep='=', drop_first=True, dtype=int)
        x, y = self.get_xy()
        # se for padronizar os dados fazer aqui (somente o x)
        self.dataset[self.class_attribute] = y