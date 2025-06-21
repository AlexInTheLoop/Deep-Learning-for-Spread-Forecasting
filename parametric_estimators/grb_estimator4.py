# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 23:16:12 2023

@author: xavie
"""

from scipy import optimize
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def objective(SE, tab_RV):
    delta = SE[0]
    S = SE[1]
    H = SE[2]
    sigma = SE[3]
    list_error = []
    for n in range(1, len(tab_RV)+1):
        pred_spread = (n**(2*H))*sigma**2 + 0.5*(S**2)*(1-np.exp(-n/delta))
        list_error.append((pred_spread - tab_RV[n-1])**2)
    sum_error = np.array(list_error).sum()
    return sum_error * (10**6)

def estimateur_grb4_close(log_price, L):
    tab_n = np.arange(1, L + 1)
    tab_RV_tot = []
    n = len(log_price)
    for L_ in tab_n:
        list_diff = np.array([log_price[i + L_] - log_price[i] for i in range(n - L_)])
        tab_RV_tot.append(np.var(list_diff))

    cons = ({'type': 'ineq', 'fun': lambda x: x[0]},
            {'type': 'ineq', 'fun': lambda x: x[1]},
            {'type': 'ineq', 'fun': lambda x: x[2]},
            {'type': 'ineq', 'fun': lambda x: x[3]})

    bnds = ((0, 1), (0, 0.1), (0, 1), (0, 1))
    x0 = np.array([
        np.random.uniform(0, 1e-4),
        np.random.uniform(0, 1e-5),
        np.random.uniform(0.1, 0.9),
        np.random.uniform(0, 1e-4)
    ])

    res = optimize.minimize(
        objective,
        x0,
        args=(tab_RV_tot,),
        method="SLSQP",
        bounds=bnds,
        constraints=cons,
        options={"maxiter": 500}
    )

    return {
        "Spread": res.x[1],
        "Hurst": res.x[2],
        "Delta": res.x[0],
        "sigma": res.x[3]
    }


### For OMC prices
def estimateur_grb4_OMC(log_open, log_close, log_mid, L):
    spread_open = estimateur_grb4_close(log_open, L)["Spread"]
    spread_close = estimateur_grb4_close(log_close, L)["Spread"]
    spread_mid = estimateur_grb4_close(log_mid, L)["Spread"]
    list_spreads = [i for i in [spread_open, spread_close, spread_mid] if i!=0]
    if len(list_spreads)!=0:
        rep = np.mean(list_spreads)
    else:
        rep=0
    return rep



##### Trying #############


# theta = 0.01  
# xi = 0.1 
# dico_sim=HF_simulation_all_auto(50, 3/100, 0.1/100, 15, 0.7, theta, xi)
# tab_spread = []
# tab_Hurst = []
# tab_sigma = []
# tab_delta = []
# tab_nit = []
# for i in dico_sim.keys():
#     df_prices = dico_sim[i]
#     tab_n=np.arange(1,10)
#     tab_RV = []
#     log_close = np.array(df_prices["Close"])
#     n=len(log_close)
#     for L in tab_n:
#         list_diff = np.array([log_close[i+L] - log_close[i] for i in range(n-L)])
#         tab_RV.append(np.var(list_diff)) 
    
    
#     cons = ({'type': 'ineq', 'fun': lambda x:  x[0]},
#         {'type': 'ineq', 'fun': lambda x:  x[1]},
#         {'type': 'ineq', 'fun': lambda x:  x[2]},
#         {'type': 'ineq', 'fun': lambda x:  x[3]})
    
#     bnds = ((0, 1), (0, 0.1) ,(0,1), (0,1))

#     x0 = np.array([np.random.normal()*10**-4, np.random.normal()*10**-4, np.random.uniform(0.1,0.9), np.random.normal()*10**-2])
       
#     res=optimize.minimize(objective, x0, method="SLSQP", bounds=bnds, constraints=cons, options={"disp":True,"maxiter":5*10**4})
#     if res.nit!=1:
#         tab_spread.append(res.x[1])
#         tab_Hurst.append(res.x[2])
#         tab_sigma.append(res.x[3])
#         tab_delta.append(res.x[0])
#         tab_nit.append(res.nit)


# theta = 0.01  
# xi = 0.1 
# dico_sim=HF_simulation_all_auto(50, 3/100, 0.75/100, 100, 0.7, theta, xi)  
# tab_spread = []
# tab_Hurst = []
# tab_sigma = []
# tab_delta = []
# tab_nit = []
# for i in dico_sim.keys():
#     df_prices = dico_sim[i]
#     log_close = np.array(df_prices["Close"])
#     rep_tot = estimateur_grb4(log_close, 10)
#     tab_spread.append(rep_tot["Spread"])
#     tab_Hurst.append(rep_tot["Hurst"])
    