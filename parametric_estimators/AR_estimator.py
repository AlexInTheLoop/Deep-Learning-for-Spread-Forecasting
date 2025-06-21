# -*- coding: utf-8 -*-
"""
Created on Wed May 31 22:38:57 2023

@author: xavie
"""

import pandas as pd
import numpy as np

def AR_estimateur(log_close, log_middle, use_oposed=False):
    eta = log_middle[:-1]
    eta_1 = log_middle[1:]
    c = log_close[:-1]
    S_2 = 4 * np.mean((c - eta) * (c - eta_1))
    rep = 0
    if use_oposed==False:
        if S_2>0:
            rep = np.sqrt(S_2)
    if use_oposed==True:
        rep = np.sqrt(np.abs(S_2))
    return rep 
    