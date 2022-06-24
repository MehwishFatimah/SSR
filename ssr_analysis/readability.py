#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 12:37:32 2021
Copied and modified on Thu Jun 16 13:10:18 2022
for ssr paper

compare spk mbart finetuned and spk mbart-st
You need base spk file as well
@author: fatimamh
env? 

readability: 'cli', 'ari' , 'lwf', 
lexical richness: 'MATTR', 'MTLD', 'HDD', 'entropy'
density: 'sentence_density', 'word_density', 'unique_word_density'
"""
import os
import pandas as pd
import numpy as np
import textstat

from lexicalrichness import LexicalRichness
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize

import collections
from scipy.stats import entropy

'''----------------------------------------------------------------
'''
def get_word_stats(text, lang):

    chars = text.replace(" ","")
    tokenized_words = word_tokenize(text, language= lang)
    
    return len(tokenized_words), len(chars)

'''----------------------------------------------------------------
'''
def get_sent_stats(text, lang):
    
    token_text = sent_tokenize(text, language=lang)
    return len(token_text)

'''----------------------------------------------------------------
'''
def estimate_shannon_entropy(text):
    bases = collections.Counter([tmp_base for tmp_base in text])
    # define distribution
    dist = [x/sum(bases.values()) for x in bases.values()]
 
    # use scipy to calculate entropy
    entropy_value = entropy(dist, base=2)
 
    return entropy_value

'''----------------------------------------------------------------
'''
def calc_lex_richness(text):
    
    lex = LexicalRichness(text)
    mattr = lex.mattr(window_size=5)
    mtld = lex.mtld(threshold=0.72)
    hdd = lex.hdd(draws=5)
    entropy = estimate_shannon_entropy(text)
    
    return mattr*100, mtld, hdd*100, entropy

'''----------------------------------------------------------------
'''
def calc_readability(text, lang):
    textstat.set_lang(lang)
    
    
    liau_index = textstat.coleman_liau_index(text)
    automated_readability = textstat.automated_readability_index(text)
    write_formula = textstat.linsear_write_formula(text)
    
    return liau_index, automated_readability, write_formula


'''----------------------------------------------------------------
'''

if __name__ == "__main__":
    
    base_file = '/hits/basement/nlp/fatimamh/inputs/wiki_t5/spk_test.csv' # text, summary
    out = '/home/fatimamh/Documents/ssr_results/statistical_features/mbart' 
    
    ft_file = '/home/fatimamh/Documents/ssr_results/fine_tuned/spk/mbart/summaries.csv' # reference, system
    ssr_file = '/home/fatimamh/Documents/ssr_results/ssr/spk/mbart/st/summaries.csv' # reference, system
    
    scores = ['cli', 'ari' , 'lwf',  
             'MATTR', 'MTLD', 'HDD', 'entropy', 
             'sent', 'word', 'char', 'sdensity', 'wdensity']
    
    group = ['text', 'gref', 'grf', 'ft', 'grs', 'ssr']
    
    columns = []
    for s in scores:
        for g in group:
            #print('{}_{}'.format(g,s))
            columns.append(g+'_'+s)
    print(columns)
    print(len(columns))

    bdf = pd.read_csv(base_file)#.head(5)
    fdf = pd.read_csv(ft_file)#.head(5)
    sdf = pd.read_csv(ssr_file)#.head(5)
    out_df = pd.DataFrame(columns = columns)
    
    # readability
    #base text
    out_df['text_cli'], out_df['text_ari'], out_df['text_lwf'] = \
    zip(*bdf['text'].apply(lambda x: calc_readability(x, 'en')))
    
    out_df['gref_cli'], out_df['gref_ari'], out_df['gref_lwf'] = \
    zip(*bdf['summary'].apply(lambda x: calc_readability(x, 'de')))
    
    #fine_tuned
    out_df['grf_cli'], out_df['grf_ari'], out_df['grf_lwf'] = \
    zip(*fdf['reference'].apply(lambda x: calc_readability(x, 'de')))
    
    out_df['ft_cli'], out_df['ft_ari'], out_df['ft_lwf'] = \
    zip(*fdf['system'].apply(lambda x: calc_readability(x, 'de')))
    
    #ssr
    out_df['grs_cli'], out_df['grs_ari'], out_df['grs_lwf'] = \
    zip(*sdf['reference'].apply(lambda x: calc_readability(x, 'de')))
    
    out_df['ssr_cli'], out_df['ssr_ari'], out_df['ssr_lwf'] = \
    zip(*sdf['system'].apply(lambda x: calc_readability(x, 'de')))
    
    
    #richness 
    #base text
    out_df['text_MATTR'], out_df['text_MTLD'], out_df['text_HDD'], out_df['text_entropy'] = \
    zip(*bdf['text'].apply(lambda x: calc_lex_richness(x)))
    
    out_df['gref_MATTR'], out_df['gref_MTLD'], out_df['gref_HDD'], out_df['gref_entropy'] = \
    zip(*bdf['summary'].apply(lambda x: calc_lex_richness(x)))
    
    #fine_tuned
    out_df['grf_MATTR'], out_df['grf_MTLD'], out_df['grf_HDD'], out_df['grf_entropy'] = \
    zip(*fdf['reference'].apply(lambda x: calc_lex_richness(x)))
    
    out_df['ft_MATTR'], out_df['ft_MTLD'], out_df['ft_HDD'], out_df['ft_entropy'] = \
    zip(*fdf['system'].apply(lambda x: calc_lex_richness(x)))
    
    #ssr
    out_df['grs_MATTR'], out_df['grs_MTLD'], out_df['grs_HDD'], out_df['grs_entropy'] = \
    zip(*sdf['reference'].apply(lambda x: calc_lex_richness(x)))
    
    out_df['ssr_MATTR'], out_df['ssr_MTLD'], out_df['ssr_HDD'], out_df['ssr_entropy'] = \
    zip(*sdf['system'].apply(lambda x: calc_lex_richness(x)))
    
    
    #density
    #base text
    out_df['text_sent'] = bdf['text'].apply(lambda x: get_sent_stats(x, 'english'))
    
    out_df['text_word'], out_df['text_char'] = \
    zip(*bdf['text'].apply(lambda x: get_word_stats(x,'english')))
    
    out_df['text_sdensity'] = out_df['text_sent'] / out_df['text_word']*100
    out_df['text_wdensity'] = out_df['text_word'] / out_df['text_char']*100
    
    out_df['gref_sent'] =  bdf['summary'].apply(lambda x: get_sent_stats(x, 'german'))
    
    out_df['gref_word'], out_df['gref_char'] = \
    zip(*bdf['summary'].apply(lambda x: get_word_stats(x,'german')))

    out_df['gref_sdensity'] = out_df['gref_sent'] / out_df['gref_word']*100
    out_df['gref_wdensity'] = out_df['gref_word'] / out_df['gref_char']*100
    
    #fine_tuned
    out_df['grf_sent'] =  fdf['reference'].apply(lambda x: get_sent_stats(x, 'german'))
    
    out_df['grf_word'], out_df['grf_char'] = \
    zip(*fdf['reference'].apply(lambda x: get_word_stats(x,'german')))

    out_df['grf_sdensity'] = out_df['grf_sent'] / out_df['grf_word']*100
    out_df['grf_wdensity'] = out_df['grf_word'] / out_df['grf_char']*100
    
    
    out_df['ft_sent'] =  fdf['system'].apply(lambda x: get_sent_stats(x, 'german'))
    
    out_df['ft_word'], out_df['ft_char'] = \
    zip(*fdf['system'].apply(lambda x: get_word_stats(x,'german')))

    out_df['ft_sdensity'] = out_df['ft_sent'] / out_df['ft_word']*100
    out_df['ft_wdensity'] = out_df['ft_word'] / out_df['ft_char']*100
    
    #ssr
    out_df['grs_sent'] =  sdf['reference'].apply(lambda x: get_sent_stats(x, 'german'))
    
    out_df['grs_word'], out_df['grs_char'] = \
    zip(*sdf['reference'].apply(lambda x: get_word_stats(x,'german')))

    out_df['grs_sdensity'] = out_df['grs_sent'] / out_df['grs_word']*100
    out_df['grs_wdensity'] = out_df['grs_word'] / out_df['grs_char']*100
    
    out_df['ssr_sent'] =  sdf['system'].apply(lambda x: get_sent_stats(x, 'german'))
    
    out_df['ssr_word'], out_df['ssr_char'] = \
    zip(*sdf['system'].apply(lambda x: get_word_stats(x,'german')))

    out_df['ssr_sdensity'] = out_df['ssr_sent'] / out_df['ssr_word']*100
    out_df['ssr_wdensity'] = out_df['ssr_word'] / out_df['ssr_char']*100
    
    
    
    print(out_df.head())
    file = os.path.join(out, 'spk_sst_ft_stats.csv')
    out_df.to_csv(file, index=False)
    