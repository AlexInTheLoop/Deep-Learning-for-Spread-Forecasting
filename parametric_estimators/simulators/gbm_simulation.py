# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 21:57:34 2022

@author: xavie
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import bernoulli

def LF_simulation(P0,sigma_month,Spread, nb_month):
    dico_prices={}
    sigma = sigma_month / np.sqrt(8190)
    for months in tqdm(range(1,nb_month)):
        df_prices=pd.DataFrame(index=np.arange(1,22),columns=["Open","High","Low","Close"])
        previous_price=P0
        for days in range(1,22):
            tab_prices=[]
            for minutes in range(390):
                price=previous_price*np.exp(sigma*np.random.normal(0,1) - 0.5*sigma**2)
                previous_price=price
                tab_prices.append(price*(1+(2*np.random.randint(2)-1)*Spread/2))
            df_prices["Open"].at[days]=tab_prices[0]
            df_prices["High"].at[days]=max(tab_prices)
            df_prices["Low"].at[days]=min(tab_prices)
            df_prices["Close"].at[days]=tab_prices[-1]
        dico_prices[months]=df_prices
    return dico_prices


def HF_simulation(P0,sigma_day,Spread,nb_days):
    dico_prices={}
    sigma = sigma_day / np.sqrt(28880)
    for days in tqdm(range(1,nb_days)):  ## (1,253) usually
        df_prices=pd.DataFrame(index=np.arange(1,481),columns=["Open","High","Low","Close"])
        previous_price=P0
        for minutes in range(1,481):
            tab_prices=[]
            for secondes in range(1,61):
                price=previous_price*np.exp(sigma*np.random.normal(0,1) - 0.5*sigma**2)
                previous_price=price
                tab_prices.append(price*(1+(2*np.random.randint(2)-1)*Spread/2))
            df_prices["Open"].at[minutes]=tab_prices[0]
            df_prices["High"].at[minutes]=max(tab_prices)
            df_prices["Low"].at[minutes]=min(tab_prices)
            df_prices["Close"].at[minutes]=tab_prices[-1]
        dico_prices[days]=df_prices
    return dico_prices

def HF_simulation_expected_trades(P0,sigma_day,Spread,nb_days, expected_trades):
    dico_prices={}
    sigma = sigma_day / np.sqrt(28880)
    for days in tqdm(range(1,nb_days)):  ## (1,253) usually
        df_prices=pd.DataFrame(index=np.arange(1,481),columns=["Open","High","Low","Close"])
        previous_price=P0
        for minutes in range(1,481):
            tab_prices=[]
            for secondes in range(1,61):
                price=previous_price*np.exp(sigma*np.random.normal(0,1) - 0.5*sigma**2)
                previous_price=price
                tab_prices.append(price*(1+(2*np.random.randint(2)-1)*Spread/2))
            array_expected = bernoulli.rvs(expected_trades, size = 60)
            tab_prices_int = []
            for idx in range(len(tab_prices)):
                if array_expected[idx]==1:
                    tab_prices_int.append(tab_prices[idx])
            if len(tab_prices_int)>0:
                df_prices["Open"].at[minutes]=tab_prices_int[0]
                df_prices["High"].at[minutes]=max(tab_prices_int)
                df_prices["Low"].at[minutes]=min(tab_prices_int)
                df_prices["Close"].at[minutes]=tab_prices_int[-1]
            else:
                if minutes>1:
                   df_prices["Open"].at[minutes] = df_prices["Open"].at[minutes-1]
                   df_prices["High"].at[minutes] = df_prices["High"].at[minutes-1]
                   df_prices["Low"].at[minutes] = df_prices["Low"].at[minutes-1]
                   df_prices["Close"].at[minutes] = df_prices["Close"].at[minutes-1]
                else:
                    df_prices["Open"].at[minutes] = P0
                    df_prices["High"].at[minutes] = P0
                    df_prices["Low"].at[minutes] = P0
                    df_prices["Close"].at[minutes] = P0
        dico_prices[days]=df_prices
    return dico_prices 

def HF_simulation_random_spread(P0,sigma_day, nb_days):
    dico_prices={}
    dico_spread = {}
    sigma = sigma_day / np.sqrt(28880)
    for days in tqdm(range(1,nb_days)):  ## (1,253) usually
        df_prices=pd.DataFrame(index=np.arange(1,481),columns=["Open","High","Low","Close"])
        previous_price=P0
        Spread = np.random.uniform(0, 1/100)
        for minutes in range(1,481):
            tab_prices=[]
            for secondes in range(1,61):
                price=previous_price*np.exp(sigma*np.random.normal(0,1) - 0.5*sigma**2)
                previous_price=price
                tab_prices.append(price*(1+(2*np.random.randint(2)-1)*Spread/2))
            df_prices["Open"].at[minutes]=tab_prices[0]
            df_prices["High"].at[minutes]=max(tab_prices)
            df_prices["Low"].at[minutes]=min(tab_prices)
            df_prices["Close"].at[minutes]=tab_prices[-1]
        dico_prices[days]=df_prices
        dico_spread[days] = Spread
    return dico_prices, dico_spread

#dico_essai, dico_spread_essai = HF_simulation_random_spread(50, 3/100, 10)