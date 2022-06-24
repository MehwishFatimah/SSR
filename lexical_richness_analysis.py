#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modified on Tue May 24 15:35:31 2022
Created on Mon Sep  6 16:34:43 2021

@author: fatimamh
env:bert-sum 
Type-token ratio (TTR) -- (Chotlos 1944, Templin 1957)
Root TTR (RTTR)  -- (Guiraud 1954, 1960) Guiraud's R and Guiraud's index
Corrected TTR (CTTR)  -- (Carrol 1964)
Herdan  -- Herdan's C. (Herdan 1960, 1964)
Summer -- (Summer 1966)
Dugast -- (Dugast 1978)
Maas -- Maas's TTR -- lower maas measure indicates higher lexical richness.
            (Maas 1972)
msttr -- Mean segmental TTR (MSTTR)
mattr -- Moving average TTR (MATTR) (Covington 2007, Covington and McFall 2010)
mtld -- Measure of textual lexical diversity (McCarthy 2005, McCarthy and Jarvis 2010)
hdd -- Hypergeometric distribution diversity (HD-D) McCarthy and Jarvis 2007)
"""

import os
import pandas as pd, numpy as np
from lexicalrichness import LexicalRichness
from nltk.tokenize import sent_tokenize
import collections
from scipy.stats import entropy


'''----------------------------------------------------------------
'''
def get_folders(root):
    folders = list(filter(lambda x: os.path.isdir(os.path.join(root, x)), os.listdir(root)))
    #print(folders)
    return folders

'''----------------------------------------------------------------
'''
def get_entropy(text):
    #print(text)
    bases = collections.Counter([tmp_base for tmp_base in text])
    # define distribution
    dist = [x/sum(bases.values()) for x in bases.values()]
    # use scipy to calculate entropy
    return entropy(dist, base=2)
 
'''----------------------------------------------------------------
'''
def get_mattr(text):
    lex = LexicalRichness(text)  
    return lex.mattr(window_size=2)*100

'''----------------------------------------------------------------
'''
def get_mtld(text):
    lex = LexicalRichness(text)
    return lex.mtld(threshold=0.72)

'''----------------------------------------------------------------
'''
def get_hdd(text):
    lex = LexicalRichness(text)
    return lex.hdd(draws=1)*100
    
'''----------------------------------------------------------------
'''
def save_hdd(root, df):
    out = os.path.join(root, 'hdd.csv')
    sdf = pd.DataFrame()
    for (name, data) in df.iteritems():
       print('Colunm Name : ', name)
       sdf[name] = df[name].apply(lambda x:get_hdd(x))
    
    print('saving file {} ...'.format(out))
    sdf.to_csv(out)

'''----------------------------------------------------------------
'''
def save_mtld(root, df):    
    out = os.path.join(root, 'mtld.csv')
    sdf = pd.DataFrame()
    for (name, data) in df.iteritems():
       print('Colunm Name : ', name)
       df[name] = df[name]
       sdf[name] = df[name].apply(lambda x:get_mtld(x))
    
    print('saving file {} ...'.format(out))
    sdf.to_csv(out)

'''----------------------------------------------------------------
'''
def save_mattr(root, df):
    out = os.path.join(root, 'mattr.csv')
    sdf = pd.DataFrame()
    for (name, data) in df.iteritems():
       print('Colunm Name : ', name)
       df[name] = df[name]
       sdf[name] = df[name].apply(lambda x:get_mattr(x))
    
    print('saving file {} ...'.format(out))
    sdf.to_csv(out)

'''----------------------------------------------------------------
'''
def save_entropy(root, df):    
    out = os.path.join(root, 'entropy.csv')
    sdf = pd.DataFrame()
    for (name, data) in df.iteritems():
       print('Colunm Name : ', name)
       df[name] = df[name]
       sdf[name] = df[name].apply(lambda x:get_entropy(x))
    
    print('saving file {} ...'.format(out))
    sdf.to_csv(out)   
    
'''----------------------------------------------------------------
'''
def get_name(f):
    name = f.split('_')
    #print(name)
    del name[:3]
    
    return ''.join(name)

'''----------------------------------------------------------------
'''
def get_list(text):
    
    sent_list = None
    if (not pd.isnull(text)) and (text is not None):
        print(text)
        print()
        sent_list = sent_tokenize(text)
        print(sent_list)
    return sent_list

'''----------------------------------------------------------------
'''
def get_combined(root, file):
    
    ref_flag = True # to add reference only once
    df = pd.DataFrame()
    folders = get_folders(root)        
    
    for f in folders:    
        folder = os.path.join(root, f)
        in_file = os.path.join(folder, 'summaries.csv')
        
        if os.path.exists(in_file):
            name = get_name(f)
            print('column name: {}'.format(name))
            temp = pd.read_csv(in_file) 
            print(temp.columns)
            
            if ref_flag: 
                df['reference'] = temp['reference']
                
                ref_flag = False
            
            if name == 'textrank':
                print('here')
                #df[name] = temp['system'].apply(lambda x: get_list(x))
            else:    
                df[name] = temp['system']
    
    print('************')
    print(df.columns)
    print(len(df))
    df.dropna
    print(len(df))
    print(df.head(5))
    df.to_csv(file, index=False)
    

'''----------------------------------------------------------------
'''
if __name__ == "__main__":

    
    root = "/home/fatimamh/Documents/ssr_results/fine_tuned/spk/mbart"
    file= os.path.join(root, 'summaries.csv')
    
    #get_combined(root, file)
    #assuming same file after get_combined
    df = pd.read_csv(file)
    print(df.columns)
    #df = df.head(5)
    
    #save_entropy(root, df)
    #save_mattr(root, df)
    #save_mtld(root, df)
    save_hdd(root, df)

#column_names = ["name", "f1", "p", "r"]
#df = df.reindex(columns=column_names)