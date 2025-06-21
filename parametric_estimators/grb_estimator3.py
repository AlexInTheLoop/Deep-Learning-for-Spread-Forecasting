import numpy as np

def estimateur_grb3_solo_close(log_price, L, use_oposed=False):
    n=len(log_price)
    diff_log_L = np.array([log_price[i+L] - log_price[i] for i in range(n-L)])
    diff_log_2L = np.array([log_price[i+2*L] - log_price[i] for i in range(n-2*L)])
    diff_log_4L = np.array([log_price[i+4*L] - log_price[i] for i in range(n-4*L)])
    V_L_delta = np.var(diff_log_L)
    V_2L_delta = np.var(diff_log_2L)
    V_4L_delta = np.var(diff_log_4L)
    num = 2 * (2 * V_L_delta - V_2L_delta)**2
    denom = (2 * np.sqrt(np.abs(2 * V_L_delta - V_2L_delta)) - np.sqrt(np.abs(2*V_2L_delta - V_4L_delta)))**2
    rep = num / denom
    if rep > 0:
        spread = np.sqrt(rep)
    else:
        if use_oposed==False:
            spread = 0
        if use_oposed==True:
            spread = np.sqrt(np.abs(rep))
    return spread 

def estimateur_grb3_close(log_price, L, use_oposed=False):
    spread_mean=[]
    for i in range(1,L+1):
        spread_mean.append(estimateur_grb3_solo_close(log_price, i, use_oposed))
    if len([i for i in spread_mean if i!=0])!=0:
        rep=np.percentile([i for i in spread_mean if i!=0],50)
    else:
        rep = 0
    return rep