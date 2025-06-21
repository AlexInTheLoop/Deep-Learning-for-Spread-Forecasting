# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 22:37:34 2022

@author: xavie
"""

import numpy as np
import math 
  

def estimateur_grb2_solo_close(log_close, n_spread, Hurst, use_oposed):
    n=len(log_close)
    diff_log=[log_close[i+1]-log_close[i] for i in range(n-1)]
    diff_logn=[log_close[i+n_spread]-log_close[i] for i in range(n-n_spread)]
    RV=np.var(diff_log)
    RVn=np.var(diff_logn)
    if 1-n_spread**(2*Hurst)!=0:
        S_est = 2*((RVn-RV*n_spread**(2*Hurst))/(1-n_spread**(2*Hurst)))
        if use_oposed == False:
            S = float(max(0,S_est)**0.5)
        else:
            S = np.sqrt(np.abs(S_est))
    else:
        S=0
    rep=0
    if Hurst>0 and Hurst<1:
        rep=S
    return rep

def estimateur_grb2_close(log_close, n_spread,Hurst, use_oposed):
    spread_mean = []
    for i in range(2,n_spread+1):
        spread_mean.append(estimateur_grb2_solo_close(log_close, i, Hurst, use_oposed))
    if len([i for i in spread_mean if i!=0])!=0:
        rep = np.percentile([i for i in spread_mean if i!=0],50)
    else:
        rep = 0        
    return rep



### For OMC prices


def estimateur_grb2_OMC(log_open, log_close, log_mid, n_spread, Hurst_open, Hurst_close, Hurst_mid, use_oposed=False):
    spread_open = estimateur_grb2_close(log_open, n_spread, Hurst_open, use_oposed)
    spread_close = estimateur_grb2_close(log_close, n_spread, Hurst_close, use_oposed)
    spread_mid = estimateur_grb2_close(log_mid, n_spread, Hurst_mid, use_oposed)
    list_spreads = [i for i in [spread_open, spread_close, spread_mid] if i!=0]
    if len(list_spreads)!=0:
        rep = np.mean(list_spreads)
    else:
        rep = 0
    return rep 
