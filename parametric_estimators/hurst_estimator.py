# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 14:45:15 2023

@author: xavie
"""
import numpy as np

def estimateur_Hurst_solo(log_close, l):
    n = len(log_close)
    tab_log_prices_dif = [log_close[i+l] - log_close[i] for i in range(n-l)]
    tab_log_prices_dif2 = [log_close[i+2*l] - log_close[i] for i in range(n-2*l)]
    tab_log_prices_dif4 = [log_close[i+4*l] - log_close[i] for i in range(n-4*l)]
    V=np.var(tab_log_prices_dif)
    V2=np.var(tab_log_prices_dif2)
    V4=np.var(tab_log_prices_dif4)
    rep = 0
    if (V2 - V) != 0 and ((V4-V2)/(V2-V)) > 0:
        rep = 0.5*np.log2((V4-V2)/(V2-V))
    return rep

def estimateur_Hurst(log_close, L):
    Hurst_med = []
    rep=0
    for l in range(1,L+1):
        Hurst_exponent = estimateur_Hurst_solo(log_close,l)
        if Hurst_exponent>0 and Hurst_exponent<1:
            Hurst_med.append(Hurst_exponent)
    if len(Hurst_med)>0:
        rep = np.percentile(Hurst_med,50)
    return rep 