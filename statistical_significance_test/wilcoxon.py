#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 17:25:36 2022

@author: fatimamh
"""
import os
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
'''----------------------------------------------------------------
'''
if __name__ == '__main__':
    
    metrics = ['rouge', 'bert', 'bleu', 'meteor', 'sari'] #'mauve', 
    embeddings = ['bm', 'ps', 'rd', 'st']
    datasets = ['wiki', 'spk']
    base_models = ['mbart', 'mt5']
    base_folder = '/home/fatimamh/Documents/ssr_results/fine_tuned' 
    new_folder = '/home/fatimamh/Documents/ssr_results/ssr'
    
    out = '/home/fatimamh/Documents/ssr_results/sst_results/wilcoxon'
    test = 'Wilcoxon'
    
    for data in datasets: 
        dataset = data
        bf = os.path.join(base_folder, data)
        nf = os.path.join(new_folder, data)
        of = os.path.join(out, data)
        print(data)
        print(bf)
        print(nf)
        print(of)
        print()
        for mod in base_models:
            model = mod
            model1 = 'SSR' 
            model2 = 'finetuned'
            
            mbf = os.path.join(bf, mod)
            mnf = os.path.join(nf, mod)
            mof = os.path.join(of, mod)
            print(mbf)
            print(mnf)
            print(mof)
            print()
            for emb in embeddings:
                embedding= emb
                enf = os.path.join(mnf, emb)
                print(enf)
                for met in metrics:
                    metric = met
                    f_name = met + '.csv'
                    base_file = os.path.join(mbf, f_name)
                    new_file = os.path.join(enf, f_name)
                    print(base_file)
                    print(new_file)
                    metric = met

                    name = metric+'_'+model1+'_'+embedding+'_'+model2+'.txt'
                    file = os.path.join(mof, name)
                    f = open(file, "w")
                    if os.path.isfile(base_file) and os.path.isfile(new_file):
                        new_df = pd.read_csv(new_file)
                        base_df = pd.read_csv(base_file)
                        stat, p = wilcoxon(new_df, base_df)
    
                        print('Statistics: {}'.format(stat))
                        print('************')
                        f.writelines('************\n')
                        
                        print('Test: {}'.format(test))
                        f.writelines('Test: {}\n'.format(test))
                        print('Metric: {}'.format(metric))
                        f.writelines('Metric: {}\n'.format(metric))
                        f.writelines('Dataset: {}\n'.format(dataset))
                        f.writelines('Model: {}\n'.format(model))
                        f.writelines('Model1-embedding: {}-{}\n'.format(model1,emb))
                        f.writelines('Model2: {}\n'.format(model2))
                        f.writelines('************\n')
                        f.writelines('\n')
                        print('Statistics = {:.4f}'.format(stat))
                        f.writelines('Statistics = {:.4f}\n'.format(stat))
                    
                        print('P-value = {:.4f}'.format(p))
                        f.writelines('P-value = {:.4f}\n'.format(p))
                        
                        print()
                        f.writelines('\n')
                        
                        alpha = 0.05
                        if p > alpha:
                            print("Same distribution (fail to reject H0)")
                            f.writelines("Same distribution (fail to reject H0)\n")
                            
                            	
                        else:
                            print("Different distribution (reject H0)")
                            f.writelines("Different distribution (reject H0)\n")
                        
                        print("{:.2f} level of significance".format(alpha))
                        f.writelines("alpha = {:.2f} level of significance.\n".format(alpha))
                        print('************')
                        f.writelines('************\n\n')
                        f.close()
                        