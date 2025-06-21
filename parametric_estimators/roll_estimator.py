import numpy as np

def Roll_estimateur(log_close, use_oposed=False):
    log_diff = [log_close[i] - log_close[i-1] for i in range(2, len(log_close))]
    log_diff_2 = [log_close[i] - log_close[i-1] for i in range(1, len(log_close)-1)]
    cov = np.cov(log_diff, log_diff_2)[0][1]
    rep = -4*cov
    if use_oposed==False:
        if rep>0:
            spread=np.sqrt(rep)
        else:
            spread = 0
    if use_oposed==True:
        spread=np.sqrt(np.abs(rep))
    return spread