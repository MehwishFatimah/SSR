#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 13:06:01 2022

@author: fatimamh
"""
import os, sys
import pandas as pd
import numpy as np
from scipy import stats

'''----------------------------------------------------------------
'''
if __name__ == '__main__':
       
    new_df = pd.read_csv('/home/fatimamh/Documents/ssr_results/ssr/wiki/mbart/bm/rouge.csv')
    base_df = pd.read_csv('/home/fatimamh/Documents/ssr_results/ssr/wiki/mt5/bm/rouge.csv')
    new = new_df['r1_f']
    base = base_df['r1_f']
    
    t_value, p_value = stats.ttest_ind(new, base)
    
    print('t_value: {}'.format(t_value))
    
    folder = '/home/fatimamh/Documents/ssr_results'
    test_name = 'wiki mbart: ssr vs ft'
    metric = 'rouge'
    name = 'wiki_mbart_rouge.txt'
    file = os.path.join(folder, name)
    f = open(file, "a+")
    
    
    print('************')
    f.writelines('************\n')
    
    print('Test: {}'.format(test_name))
    f.writelines('Test: {}\n'.format(test_name))
    
    print('Metric: {}'.format(metric))
    f.writelines('Metric: {}\n'.format(metric))
    
    print('Test statistic = {:.4f}'.format(t_value))
    f.writelines('Test statistic = {:.4f}\n'.format(t_value))
    
    print('p-value for two tailed test = {:.4f}'.format(p_value))
    f.writelines('p-value for two tailed test = {:.4f}\n'.format(p_value))
    
    print()
    f.writelines('\n')
    
    alpha = 0.05
    if p_value <= alpha:
        print('Conclusion:\np-value={:.4f} > alpha= {:.4f}'.format(p_value,alpha))
        f.writelines('Conclusion:\np-value={:.4f} > alpha= {:.4f}\n'.format(p_value,alpha))
        print("We reject the null hypothesis H0=(no effect). Means there is significant difference")
        f.writelines("We reject the null hypothesis H0=(no effect). Means there is a significant difference.\n")
        print("μ1 = μ2 at {:.2f} level of significance".format(alpha))
        f.writelines("μ1 = μ2 at {:.2f} level of significance.\n".format(alpha))	
    else:
        print('Conclusion:\np-value={:.4f} < alpha= {:.4f}'.format(p_value,alpha))
        f.writelines('Conclusion:\np-value={:.4f} < alpha= {:.4f}\n'.format(p_value,alpha))
        print("We fail to reject the null hypothesis H0=(no effect). Means there is no significant difference.")
        f.writelines("We fail to reject the null hypothesis H0=(no effect). Means there is no significant difference.\n")
    print('************')
    f.writelines('************\n\n')
    f.close()