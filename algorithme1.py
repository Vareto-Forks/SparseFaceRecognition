#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 16:31:57 2017

@author: maxwab
"""

import numpy as np
from PIL import Image
import os, sys
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from features_reduction import *
import time
import random
import multiprocessing
from multiprocessing import Pool
import functools

def evaluate_coeff(y,Xtrain,ytrain, clf):
    clf.fit(Xtrain,y)
    x = clf.coef_
    # We make the prediction
    a = residu(y,Xtrain,x,ytrain)
    b = np.argmin(a)
    c = SCI(x,ytrain)
    return [a,b,c]

def delta(x,i,classs):
    '''
    fonction indicatrice de la classe i
    '''
    n,m = len(x),len(classs)
    
    if (n != m):
        print ('vectors of differents sizes, cannot operate delta')
        
    tmp = i*np.ones(n)-classs

    for k in range(n):
        if tmp[k]==0:
            tmp[k]=1
        else:
            tmp[k]=0 
            
    return tmp*x


# Definition of the residue function that returns the class minimizing the reconstruction error according to the L2 norm.

def residu(y,A,x,class_x):
    '''
    renvoie les residus pour chaque classe.
    '''
    k = np.max(class_x)+1
    r = np.zeros(k)
    
    for i in range(0,k):
        r[i] = np.linalg.norm(y - np.dot(A,delta(x,i,class_x)))
        
    return r


# Definition of the function returning the concentration index (_Sparsity Concentration Index_)


def SCI(x,classs):
    '''
    @input
      - classs: classe de chaque training element.
      - x     : sparse coefficients
    '''
    
    k = len(set(classs)) # Number of different classes
    
    # Puis on retourne la valeur du SCI
    return (k*(1/np.linalg.norm(x,ord=1))*np.max([np.linalg.norm(delta(x,i,classs),ord=1) for i in range(k)]) - 1)/(k-1)
    


# ** Utility **: find the class of a test set item for the Yale Database


def find_class(i):
    return int(i)/12

def noise_image(image_input, per=0.5):
    '''
    Ajoute du bruit aléatoire à une image.
    @params:
        per : pourcentage de pixels à corrompre
    '''
    sz0 = image_input.shape[0]
    sz1 = image_input.shape[1]
    
    # Mask creation
    nb_pix_to_noise = int(np.floor(per*sz0*sz1))
    
    mask = np.ones((sz0*sz1,1))
    ids = np.random.permutation(sz0*sz1)[0:nb_pix_to_noise]
    mask[ids] = 0
    mask = np.reshape(mask,(sz0,sz1))
    
    # Random pixel matrix (intensity between 0 and 255)
    rand_pix = np.random.randint(0,256,size=(sz0,sz1))
    
    # We return the initial matrix where we changed the pixels indicated by random pixels
    return np.multiply(mask,image_input) + np.multiply(1-mask,rand_pix)
    
def black_frame(array_orig, x0, x1, y0, y1):
    '''
    Ajoute un bandeau noir dont les coins sont de coordonnées (x0,y0),(x1,y1),(x0,y1) et (x1,y0)
    '''
    array_tmp = np.ones_like(array_orig)
    
    if ((x0 > array_tmp.shape[1]) | (x1 > array_tmp.shape[1]) | (y0 > array_tmp.shape[0]) | (y1 > array_tmp.shape[0])):
        print ("Error : coordonnees du bandeau trop grandes pour l'image")
    
    for i in range(array_tmp.shape[1]):
        for j in range(array_tmp.shape[0]):
            if ((i >= min(x0,x1)) & (i <= max(x0,x1)) & (j >= min(y0,y1)) & (j <= max(y0,y1))):
                array_tmp[j][i] = 0
    
    return array_tmp*array_orig
    
def matrix_transform(X):
    '''
    Ici X est une liste de np.array (sous forme de matrice, pas de vecteur)
    '''
    
    X_toconcat = [np.reshape(e,(X[0].shape[0]*X[0].shape[1],1)) for e in X]
    
    # Then concatenate to have a unique matrix
    return np.concatenate(X_toconcat,axis=1) # List of the samples of the train, concatenated in column.
    
    
###### Algorithme
    

def SRC(Xtrain, Xtest, ytrain, type_feature_reduc=None, reduce_lines=12, reduce_columns=10, lambda_val=0.02, per_bruit=0.3, pos_occl=None):
    '''
    @params :
        * Xtrain : iterable of numpy arrays representing faces
        * Xtest : iterable of numpy arrays representing faces
        * per_bruit : pourcentage de pixels à remplacer par du bruit uniforme sur [0,255]
        * pos_occl : position du rectangle pour occlure (xgauche,yhaut,xdroite,ybas)
    '''
    
    # ---- Define the parameters
    
    n_train = len(Xtrain)
    n_test = len(Xtest)
    
    n_components = reduce_lines*reduce_columns
    
    k = np.max(ytrain)+1 #Nb de classes
    
    
    s0 = Xtrain[0].shape[0]
    s1 = Xtrain[0].shape[1]



    # ---- At first we noised the test set if necessary
    
    if (per_bruit != None):
        tmp = np.copy(Xtest)
        Xtest = [noise_image(e,per=per_bruit) for e in tmp]

    # ---- Then we corrupts the test set if necessary with an occlusion
    
    if (pos_occl != None):
        xgauche,yhaut,xdroite,ybas = pos_occl[0],pos_occl[1],pos_occl[2],pos_occl[3]
        tmp = np.copy(Xtest)
        Xtest = [black_frame(e,xgauche,xdroite,ybas,yhaut) for e in tmp]
                 
    # ---- Transformation into two matrices rather than two matrices lists
    
    Xtrain = matrix_transform(Xtrain)
    Xtest = matrix_transform(Xtest)
    
                 
    # ---- Normalization
    
    ss = StandardScaler()
    # Note: we normalize the two separations because we are just taking each picture back to a unit length following the norm 2
    Xtrain = ss.fit_transform(Xtrain)
    Xtest = ss.fit_transform(Xtest)

        
    # ---- Then we do a dimension reduction for both
    
    # Several cases:
    # - None (classic): we just resize with nearest
    # - fisherfaces
    # - randomfaces
    # - eigenfaces
    
    
    # Note: we do not forget to transpose to be able to use features_reduction functions

    if (type_feature_reduc == 'eigenfaces'):
        Xtrain, Xtest = eigenfaces(Xtrain.T,Xtest.T,n_components=n_components)
        Xtrain, Xtest = Xtrain.T, Xtest.T
    elif (type_feature_reduc == 'fisherfaces'):
        Xtrain, Xtest = fisherfaces(Xtrain.T,ytrain,Xtest.T,n_components=n_components)
        Xtrain, Xtest = Xtrain.T, Xtest.T
    elif (type_feature_reduc == 'randomfaces'):
        Xtrain, Xtest = randomfaces(Xtrain.T, Xtest.T, n_components=n_components)
        Xtrain, Xtest = Xtrain.T, Xtest.T
    else: # Classic case
         # We need to reuse PIL here ... so reshape again each column and have a list of items!
        List_Xtrain, List_Xtest = [], []
    
        for j in range(Xtrain.shape[1]):
            tmp = np.reshape(Xtrain[:,j],(s0,s1))
            im = Image.fromarray(tmp)
            im = im.resize((reduce_lines,reduce_columns), Image.NEAREST) # Values indicated in parameter of the function
            List_Xtrain.append(np.asarray(im, dtype=np.float64))
        Xtrain = matrix_transform(List_Xtrain)
            
        for j in range(Xtest.shape[1]):
            tmp = np.reshape(Xtest[:,j],(s0,s1))
            im = Image.fromarray(tmp)
            im = im.resize((reduce_lines,reduce_columns), Image.NEAREST) # Values indicated in parameter of the function 
            List_Xtest.append(np.asarray(im, dtype=np.float64))
        Xtest = matrix_transform(List_Xtest)
    
        
        
    # ---- Then we apply the Lasso minimization for each example of the test set
    
    # Recall :
    # * y: Element to test (is a column of Xtest)
    # * Xtrain: Matrix A examples of training
    # * x: coefficients from LASSO minimization
    # * ytrain: class training examples

    
    preds = np.zeros(Xtest.shape[1])
    rejections = np.zeros(Xtest.shape[1])
    residus = np.zeros((k,Xtest.shape[1]))


    # we create a Lasso classifier with the specified lambda parameter
    clf = Lasso(alpha=lambda_val) 
    
    L_y = []

    for j in range(Xtest.shape[1]):
        L_y.append(Xtest[:,j]) # Current example to test
    
    nbCores = multiprocessing.cpu_count()
    pool = Pool(nbCores)
    L = pool.map(functools.partial(evaluate_coeff,Xtrain=Xtrain, ytrain=ytrain,clf=clf), L_y)
    pool.close()
    pool.join()

      
    
    
# For each example to test we generate the coefficients and we take the best prediction
# for j in range (Xtest.shape [1]):
# y = Xtest [:, j] # Current example to test
# clf.fit (Xtrain, y)
# x = clf.coef_
#
# # We make the prediction
# residus [:, j] = residu (y, Xtrain, x, ytrain)
# preds [j] = np.argmin (residual (y, xtrain, x, ytrain))
# rejections [j] = SCI (x, ytrain)
    

    # ---- We return the values above

    # predictions: vector predictions for each example of the test set
    # sci: ICS vector for each element of the test set
    # residues: residue matrix: m rows (number of classes), n columns (number of elements of the set test)
    return L
    #return preds, rejections, residues