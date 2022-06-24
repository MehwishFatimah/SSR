#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 15:06:36 2021
Copied and modified on Thu Jun 16 13:10:18 2022
for ssr paper

compare spk mbart finetuned and spk mbart-st

@author: fatimamh


"""
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

labels_dict = {'sent': 'Sentence Count', 
               'word': 'Word Count', 
               'char': 'Character Count', 
               'sdensity': 'Sentence Density', 
               'wdensity': 'Word Density', 
               'MATTR':'Moving average TTR (MATTR) - Window=5', 
               'MTLD': 'Measure of Textual Lexical Diversity (MTLD) - Threshold=0.72', 
               'HDD': 'Hypergeometric Distribution Diversity (HDD) - Draws=5', 
               'entropy': 'Shannon Entropy',
               'cli': 'Coleman Liau Index',
               'lwf': 'Linsear Write Formula',
               'ari': 'Automated Readability',}


'''----------------------------------------------------------------
''' 
def box_plot(df, name):
    
    sns.set(font_scale = 2)
    
    sns.set_theme(style="ticks")
    plt.figure(figsize=(8,6))
    plt_labels = ['Text', 'Gold', 'FT', 'SSR']#list(set(df['label']))
    #print(plt_labels)
    fig = sns.boxplot(x=name, y= 'label', data = df, hue='label', palette="husl",dodge=False,)#, width=2) 
    #fig = sns.boxplot(x='label', y= name , data = df, hue='label', palette="husl",dodge=False,)#, width=2) 
    
    fig.set(ylabel=None)
    fig.set(yticklabels=plt_labels)
    
    fig.set(xlabel=labels_dict[name])
    fig.legend_.set_title(None)
    
    plt.legend(loc=0)
    f_name = 'box_' + name + '.jpg'
    file = os.path.join(folder, f_name)
    
    plt.savefig(file, dpi=600)
    plt.show()
    
    
'''----------------------------------------------------------------
''' 
def violin_plot(df, name):
    
    print(name)
    #if name == 'ari':
        #del df[] 
    sns.set(font_scale = 2)
    sns.set_theme(style="ticks")
    plt_labels = ['Text', 'Gold', 'FT', 'SSR']
    f, ax = plt.subplots(figsize=(8,6))
    #fig = sns.violinplot(x=name, y= 'label', hue='label', data = df, palette="husl", orient = 'h', scale="width",dodge=False)#width=1.15)
    fig = sns.violinplot(x='label', y= name, hue='label', data = df, palette="husl", scale="width",dodge=False,aspect = 1.5,
                       legend_out=True)#width=1.15)
    
    fig.set(xlabel=None)
    fig.set(xticklabels=plt_labels )
    
    #fig.set(yticklabels=[name])
    fig.set(ylabel=labels_dict[name])
    fig.legend_.set_title(None)
    
    plt.legend(loc=0)
    if name == 'cli':
        plt.legend(loc=2)
    if name == 'MTLD':  
        plt.legend(loc=2)
    if name == 'sdensity':
        plt.legend(loc=2)
    if name == 'lwf':
        plt.legend(loc=3)
    if name == 'entropy':  
        plt.legend(loc=3)
    if name == 'wdensity':
        plt.legend(loc=3)
    
    
    #fig._legend.remove()
    f_name = 'violin_' + name + '.jpg'
    file = os.path.join(folder, f_name)
    plt.savefig(file, dpi=600)
    plt.show() 
    
'''----------------------------------------------------------------
'''
def plot_stats(df, pairs):
    
    
    for pair in pairs:
        print(pair[0])
        name = pair[0].split('_',1)[1]
        print(name)
        
        text = df[pair[0]].to_list()
        gr = df[pair[1]].to_list()
        ft = df[pair[3]].to_list()
        ssr = df[pair[5]].to_list()
        
        #srf = df[pair[4]].to_list()
        #grf = df[pair[2]].to_list()
        
        joined_list = text + gr + ft + ssr #+ grf+ srf 
        #print(joined_list)
        
        text_l = ['Text']*len(text)
        gr_l = ['Gold']*len(gr)
        ft_l = ['FT']*len(ft)
        ssr_l = ['SSR']*len(ssr)
        
        #grf_l = ['grf']*len(grf)
        #srf_l = ['srft']*len(srf)
        
        joined_labels = text_l + gr_l +ft_l + ssr_l #+ grf_l + srf_l 
        #print(joined_labels)
        
        ndf = pd.DataFrame(columns={name, 'label'})
        ndf[name] = joined_list
        ndf['label'] = joined_labels

        box_plot(ndf, name)
        violin_plot(ndf, name)
    

'''----------------------------------------------------------------
'''
def plot_lex(df, pairs):
    
    for pair in pairs:
        print(pair[0])
        name = pair[0].split('_',1)[1]
        print(name)
        
        text = df[pair[0]].to_list()
        gr = df[pair[1]].to_list()
        ft = df[pair[3]].to_list()
        ssr = df[pair[5]].to_list()
        
        #srf = df[pair[4]].to_list()
        #grf = df[pair[2]].to_list()
        
        joined_list = text + gr + ft + ssr #+ grf+ srf 
        #print(joined_list)
        
        text_l = ['Text']*len(text)
        gr_l = ['Gold']*len(gr)
        ft_l = ['FT']*len(ft)
        ssr_l = ['SSR']*len(ssr)
        
        #grf_l = ['grf']*len(grf)
        #srf_l = ['srft']*len(srf)
        
        joined_labels = text_l + gr_l +ft_l + ssr_l #+ grf_l + srf_l 
        #print(joined_labels)
        
        ndf = pd.DataFrame(columns={name, 'label'})
        ndf[name] = joined_list
        ndf['label'] = joined_labels

        box_plot(ndf, name)
        violin_plot(ndf, name)
    

'''----------------------------------------------------------------
'''
def plot_read(df, pairs):        
   
    for pair in pairs:
        print(pair[0])
        name = pair[0].split('_',1)[1]
        print(name)
        
        text = df[pair[0]].to_list()
        gr = df[pair[1]].to_list()
        ft = df[pair[3]].to_list()
        ssr = df[pair[5]].to_list()
        
        #srf = df[pair[4]].to_list()
        #grf = df[pair[2]].to_list()
        
        joined_list = text + gr + ft + ssr #+ grf+ srf 
        #print(joined_list)
        
        text_l = ['Text']*len(text)
        gr_l = ['Gold']*len(gr)
        ft_l = ['FT']*len(ft)
        ssr_l = ['SSR']*len(ssr)
        
        #grf_l = ['grf']*len(grf)
        #srf_l = ['srft']*len(srf)
        
        joined_labels = text_l + gr_l +ft_l + ssr_l #+ grf_l + srf_l 
        #print(joined_labels)
        
        ndf = pd.DataFrame(columns={name, 'label'})
        ndf[name] = joined_list
        ndf['label'] = joined_labels

        box_plot(ndf, name)
        violin_plot(ndf, name)


'''----------------------------------------------------------------
'''
if __name__ == "__main__":
    folder = '/home/fatimamh/Documents/ssr_results/statistical_features/mbart'
    
    file = os.path.join(folder, 'spk_ssr_ft_stats.csv')    
    df = pd.read_csv(file)#.head(20)
        
    group = ['text', 'gref', 'grf', 'ft', 'grs', 'ssr']       
    read = ['cli', 'ari' , 'lwf']
    lex = ['MATTR', 'MTLD', 'HDD', 'entropy']
    stat = ['sent', 'word', 'char', 'sdensity', 'wdensity']

    
    read_pairs = []
    for s in read:
        p = []
        for g in group:
            #print('{}_{}'.format(g,s))
            p.append(g+'_'+s)
        read_pairs.append(p)
    #print(read_pairs)
    #print(len(read_pairs))
    
    plot_read(df, read_pairs)
    
    stat_pairs = []
    for s in stat:
        p = []
        for g in group:
            #print('{}_{}'.format(g,s))
            p.append(g+'_'+s)
        stat_pairs.append(p)
    #print(stat_pairs)
    #print(len(stat_pairs))
    plot_stats(df, stat_pairs)
    
    lex_pairs = []
    for s in lex:
        p = []
        for g in group:
            #print('{}_{}'.format(g,s))
            p.append(g+'_'+s)
        lex_pairs.append(p)
    #print(lex_pairs)
    #print(len(lex_pairs))

    plot_lex(df,lex_pairs)
   
   
    
    