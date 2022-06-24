#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 13:51:19 2022

@author: fatimamh
Created Heatmaps for all automatic metrics and shallow features
"""
import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as mp
import seaborn as sb

'''----------------------------------------------------------------
'''
if __name__ == '__main__':
       
       #df1=pd.read_csv('/home/fatimamh/Documents/ssr_results/fine_tuned/spk/mbart/rouge.csv')
       #df2=pd.read_csv('/home/fatimamh/Documents/ssr_results/fine_tuned/spk/mbart/bert.csv')
       #df3=pd.read_csv('/home/fatimamh/Documents/ssr_results/fine_tuned/spk/mbart/sari.csv')
       
       df1=pd.read_csv('/home/fatimamh/Documents/ssr_results/ssr/wiki/mbart/st/rouge.csv')
       df2=pd.read_csv('/home/fatimamh/Documents/ssr_results/ssr/wiki/mbart/st/bert.csv')
       df3=pd.read_csv('/home/fatimamh/Documents/ssr_results/ssr/wiki/mbart/st/sari.csv')
       
       df4=pd.read_csv('/home/fatimamh/Documents/ssr_results/statistical_features/mbart/spk_ssr_ft_stats.csv')
       
       data = pd.DataFrame()
       data['R1_F'] = df1['r1_f']
       #data['R1_P'] = df1['r1_p']
       #data['R1_R'] = df1['r1_r']
       
       data['R2_F'] = df1['r2_f']
       #data['R2_P'] = df1['r2_p']
       #data['R2_R'] = df1['r2_r']
       
       data['RL_F'] = df1['rL_f']
       #data['RL_P'] = df1['rL_p']
       #data['RL_R'] = df1['rL_r']
       
       data['BS_F'] = df2['bs_f']
       #data['BS_P'] = df2['bs_p']
       #data['BS_R'] = df2['bs_r']
       
       data['SI'] = df3['sari']
       
       
       """
       data['CLI'] = df4['ft_cli']
       data['LWF'] = df4['ft_lwf']
       
       data['MTLD'] = df4['ft_MTLD']
       data['HDD'] = df4['ft_HDD']
       
       data['SD'] = df4['ft_sdensity']
       data['WD'] = df4['ft_wdensity']
       """
       
       data['CLI'] = df4['ssr_cli']
       data['LWF'] = df4['ssr_lwf']
       
       data['MTLD'] = df4['ssr_MTLD']
       data['HDD'] = df4['ssr_HDD']
       
       data['SD'] = df4['ssr_sdensity']
       data['WD'] = df4['ssr_wdensity']
       
       
       
       
       sb.set_theme(style='white')
       sb.set(rc = {'figure.figsize':(15,8)})
       
       
       # plotting correlation heatmap
       #dataplot=sb.heatmap(data.corr(), annot=True, cmap="Blues")
       # displaying heatmap
       #mp.show()
       
       # creating mask
       mask = np.triu(np.ones_like(data.corr()))
        
       # plotting a triangle correlation heatmap
       dataplot = sb.heatmap(data.corr(), cmap="Blues", annot=True, mask=mask)
       
       #name = 'stat_fine-tuned_mb_spk.jpg'
       name = 'stat_ssr_mb-st_spk.jpg'
       file = os.path.join('/home/fatimamh/Documents/ssr_results/heat_maps', name)
       mp.savefig(file, dpi=600)
       # displaying heatmap
       mp.show()
       
       
      
       
      
