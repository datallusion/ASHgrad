# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 10:25:24 2021

@author: Jared
"""

import matplotlib.pyplot as plt
import os
import xlrd
import pandas as pd
import numpy as np


root_dir=r'??' #path to the data file
DETERM_runs=xlrd.open_workbook(os.path.join(root_dir,'SO_runs.xls'))
sheets=[]
runs=[]
for i in range(0,5):
    sheets.append(DETERM_runs.sheet_by_index(i))
    SGD_h=sheets[i].col_values(4)[1:]
    SGD_l=sheets[i].col_values(9)[1:]
    SO=sheets[i].col_values(14)[1:]
    ADAM=sheets[i].col_values(19)[1:]
    if i==1:
        ADAM=sheets[i].col_values(24)[1:]
    index=range(0,len(SGD_h))
    run_results=np.array((index,SGD_h,SGD_l,SO,ADAM)).T
    runs.append(run_results)
    
    plt.plot(runs[i][:,0],runs[i][:,1],color='tab:orange',label='SGD High', ls='-.')
    plt.plot(runs[i][:,0],runs[i][:,2],color='tab:blue', label='SGD Low' ,ls='--')
    plt.plot(runs[i][:,0],runs[i][:,3],color='tab:red', label='ASHgrad' ,ls='-')
    plt.plot(runs[i][:,0],runs[i][:,4],color='tab:green', label='ADAM' ,ls=':')
    
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    if i ==0:
        plt.title('Small net CIFAR10')
        plt.ylim((0,100))
    elif i == 1:
        plt.title('ResNet50 CIFAR10')
        plt.ylim((0,70))
    elif i == 2:
        plt.title('EfficientNet B1 CIFAR10')
        plt.ylim((0,80))
    elif i == 3:
        plt.title('EfficientNet B2 CIFAR10')  
        plt.ylim((0,80))
    elif i == 4:
        plt.title('EfficientNet B1 SVHN Cropped')
        plt.ylim((0,100))
    plt.legend(loc='lower right')
    #plt.show()
    loc=os.path.join(root_dir,'SO_fig'+str(i+1)+'.png')
    plt.savefig(loc, format='png')
    plt.clf()


