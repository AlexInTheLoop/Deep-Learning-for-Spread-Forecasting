# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 21:03:03 2023

@author: xavie
"""
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt


def Ornstein_Uhlenbeck(length, theta, xi):
    rep = [0]
    for i in range(1,length):
        rep.append(rep[i-1] - theta*rep[i-1] + xi*np.random.normal())
    return rep 

def HF_simulation_auto_spread(P0, sigma_day, Spread, theta, xi, nb_days): 
    dico_prices={}
    sigma = sigma_day / np.sqrt(28880)
    for days in tqdm(range(1,nb_days)):  
        df_prices=pd.DataFrame(index=np.arange(1,481),columns=["Open","High","Low","Close"])
        previous_price=P0
        OU_process = Ornstein_Uhlenbeck(28880, theta, xi)
        compt=0
        for minutes in range(1,481):
            tab_prices=[]
            for secondes in range(1,61):
                price=previous_price*np.exp(sigma*np.random.normal(0,1) - 0.5*sigma**2)
                previous_price=price
                tab_prices.append(price*(1+0.5*Spread*np.sign(OU_process[compt])))
                compt+=1
            df_prices["Open"].at[minutes]=tab_prices[0]
            df_prices["High"].at[minutes]=max(tab_prices)
            df_prices["Low"].at[minutes]=min(tab_prices)
            df_prices["Close"].at[minutes]=tab_prices[-1]
        dico_prices[days]=df_prices
    return dico_prices

def LF_simulation_auto_spread(P0, sigma_month, Spread, theta, xi, nb_month): 
    dico_prices = {}
    sigma = sigma_month / np.sqrt(8190)
    for month in tqdm(range(1,nb_month)):  
        df_prices=pd.DataFrame(index=np.arange(1,22),columns=["Open","High","Low","Close"])
        previous_price=P0
        OU_process = Ornstein_Uhlenbeck(8190, theta, xi)
        compt=0
        for days in range(1,22):
            tab_prices=[]
            for minutes in range(390):
                price=previous_price*np.exp(sigma*np.random.normal(0,1) - 0.5*sigma**2)
                previous_price=price
                tab_prices.append(price*(1+0.5*Spread*np.sign(OU_process[compt])))
                compt+=1
            df_prices["Open"].at[days]=tab_prices[0]
            df_prices["High"].at[days]=max(tab_prices)
            df_prices["Low"].at[days]=min(tab_prices)
            df_prices["Close"].at[days]=tab_prices[-1]
        dico_prices[month]=df_prices
    return dico_prices


# theta = 0.01     
# xi = 0.1  
# array_OU = Ornstein_Uhlenbeck(28880, theta, xi)
# plt.plot(array_OU)
# plt.title(str(theta)+" "+str(xi))
# plt.show()
# print(len([i for i in array_OU if i>0]))
# print(len([i for i in array_OU if i<0]))
# array_prices = HF_simulation_auto_spread(50, 3/100, 0.001, theta, xi, 2)
# plt.plot(array_prices[1]["Close"])
# plt.title(str(theta)+" "+str(xi))
# plt.show()