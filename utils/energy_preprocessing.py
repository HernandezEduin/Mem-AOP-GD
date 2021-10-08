# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 19:46:31 2021

@author: Eduin Hernandez
"""
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

def prepare_data(dataset: np.ndarray) -> [np.ndarray, np.ndarray]:
    features = np.concatenate((dataset[:, 0:5], dataset[:, 6][:, np.newaxis]), axis=1)
    feature2 = np.array(dataset[:,5], dtype=int)[:, np.newaxis]
    feature3 = np.array(dataset[:,7], dtype=int)[:, np.newaxis]
    feature2 -= feature2.min()
    
    feature2 = OneHotEncoder().fit_transform(feature2).toarray()
    feature3 = OneHotEncoder().fit_transform(feature3).toarray()
    
    input_features = np.concatenate((features, feature2, feature3), axis=1)
    output_features = dataset[:,8][:,np.newaxis]
    return input_features, output_features

def split(x: np.ndarray, y: np.ndarray, split: float = 0.75) -> [np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    assert split > 0 and split < 1, 'Split percentage must be below 1 and higher than 0'
    size = x.shape[0]
    train_num = int(size*split)
    return x[:train_num], y[:train_num], x[train_num:], y[train_num:]  

def normalize(x_train: np.ndarray, x_test: np.ndarray) -> [np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    scaler.fit(x_train[:, :6])
    x_train = np.concatenate((scaler.transform(x_train[:, :6]), x_train[:, 6:]), axis=1)
    x_test = np.concatenate((scaler.transform(x_test[:, :6]), x_test[:, 6:]), axis=1)
    return x_train, x_test