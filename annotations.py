#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 11:12:12 2022
This code also generate rating table 
https://www.lighttag.io/blog/krippendorffs-alpha/

@author: fatimamh
"""
from __future__ import division
import os
import pandas as pd
import numpy as np

from statsmodels.stats.inter_rater import fleiss_kappa
import krippendorff

import kendall_w as kw
'''----------------------------------------------------------------
'''
def column_agreement(c1, c2, c3, c4, c5, raters):
    powers = pow(c1,2)+pow(c2,2)+pow(c3,2)+pow(c4,2)+pow(c5,2)
    print(powers)
    powers = powers -raters
    print(powers)
    powers = powers / (raters*(raters-1))
    print(powers)
    return ((pow(c1,2)+pow(c2,2)+pow(c3,2)+pow(c4,2)+pow(c5,2))-raters)/(raters*(raters-1))

'''----------------------------------------------------------------
'''

def mfleiss_kappa(df, raters):
    
    n_item, n_cat =  df.shape
    
    print('cat: {}'.format(n_item))
    print('rater: {}'.format(raters))
    
    df['new_col'] = df.apply(lambda x: column_agreement(x.worst, x.bad, x.neutral, x.good, x.best, raters), axis=1)
    
    print(df)
    
    kappa = ''
    return kappa

'''----------------------------------------------------------------
'''
if __name__ == '__main__':
    
    cat_mapping = {1:'worst',
                   2:'bad',
                   3:'neutral',
                   4:'good',
                   5:'best'}
    files = ['annotations_1.csv', 'annotations_2.csv', 'annotations_3.csv', 'annotations_4.csv'] # 4 annotators
    main = '/home/fatimamh/Documents/ssr_human_annotations/annotations' 
    cats = [1, 2, 3, 4, 5] # I have 5 categories 
    cat_name = ['worst', 'bad', 'neutral', 'good', 'best']
    items = 20
    param = ['s_s', 's_f', 's_r', 's_o', 'f_s', 'f_f', 'f_r', 'f_o'] #4 param and 2 models
    out = '/home/fatimamh/Documents/ssr_human_annotations/annotations/out' 
    """
    f1 = os.path.join(main, 'annotations_1.csv')
    f2 = os.path.join(main, 'annotations_2.csv')
    
    f3 = os.path.join(main, 'annotations_3.csv')
    f4 = os.path.join(main, 'annotations_4.csv')
    
    a1=pd.read_csv(f1)
    a2=pd.read_csv(f2)
    a3=pd.read_csv(f3)
    a4=pd.read_csv(f4)
    
    #separate the parameters
    
    for p in param:
        print(p)
        df = pd.DataFrame(columns = ['a1','a2','a3','a4'])
        df['a1'] = a1[p]
        df['a2'] = a2[p]
        df['a3'] = a3[p]
        df['a4'] = a4[p]
        name = p+'.csv'
        file = os.path.join(out,name)
        df.to_csv(file, index=False)
    
    for p in param:
        name = p+'.csv'
        print(name)
        out_name = p+'_cm.csv'
        print(out_name)
        file = os.path.join(out,name)
        out_file = os.path.join(out,out_name)
        
        df = pd.read_csv(file)
        #print(df)
        df = df.transpose()
        print(df)
        count_col = df.shape[1]
        print(count_col)
        out_df = pd.DataFrame(columns = cat_name)
        row = {}
        for i in range(count_col):
            print('i: {} df[i]:\n{}'.format(i, df[i]))
            row = {}
            for c in cats:
                col = cat_mapping[c]
                print('c: {}, col: {}'.format(c, col))
                count =len(df[df[i].map(lambda x: x==c)])
                print('count: {}'.format(count))
                row[col]= count
            print(row)
            out_df = out_df.append(row, ignore_index=True)
        print(out_df)
        out_df.to_csv(out_file, index=False)
        print() 
    """
    for p in param:
        name =  p+'.csv' #'_cm.csv'
        print('**********************')
        print(name)
        file = os.path.join(out, name)
        df = pd.read_csv(file)
        #k = fleiss_kappa(df)#, 4)
        #print(k)
        a = krippendorff.alpha(df)
        print(a)
        vals = df.values
        vals = vals.tolist()
        W = kw.compute_w(vals)
        print(W)
        break
    
    