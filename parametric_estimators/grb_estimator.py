# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 17:35:12 2023

@author: xavie
"""


import numpy as np 

## For close prices only 
def estimateur_grb1_solo_close(log_close, n_spread, use_oposed):
    n=len(log_close)
    diff_logc = [log_close[i+1]-log_close[i] for i in range(n-1)]
    diff_lognc=[log_close[i+n_spread]-log_close[i] for i in range(n-n_spread)]
    RVc=np.var(diff_logc)
    RVnc=np.var(diff_lognc)
    Sc = (2 / (1-n_spread)) * (RVnc - n_spread*RVc)
    if use_oposed == False:
        rep = float(max(0, Sc) ** 0.5)
    else:
        rep = np.sqrt(np.abs(Sc))
    return rep
    

def estimateur_grb1_close(log_close, n_spread, use_oposed):
    spread_mean=[]
    for i in range(2,n_spread+1):
        spread_mean.append(estimateur_grb1_solo_close(log_close, i, use_oposed))
    if len([i for i in spread_mean if i!=0])!=0:
        rep = np.percentile([i for i in spread_mean if i!=0],50)
    else:
        rep = 0        
    return rep


### For OMC prices

def estimateur_grb1_OMC(log_close, log_open, log_mid, n_spread, use_oposed):
    spread_open = estimateur_grb1_close(log_open, n_spread, use_oposed)
    spread_close = estimateur_grb1_close(log_close, n_spread, use_oposed)
    spread_mid = estimateur_grb1_close(log_mid, n_spread, use_oposed)
    list_spreads = [i for i in [spread_open, spread_close, spread_mid] if i!=0]
    if len(list_spreads)!=0:
        rep = np.mean(list_spreads)
    else:
        rep = 0
    return rep 