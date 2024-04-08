# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 13:53:14 2024

@author: Jiaqi
"""

import numpy as np
from scipy import stats
import pandas as pd

def equity(St,r,q,vol,delta,Z):
    Stdelta = St * np.exp((r-q-0.5*vol**2)*delta+vol*Z*np.sqrt(delta))
    return Stdelta

def correlated_random_values(size,rho):
    random1 = np.random.normal(size=size)
    random2 = np.random.normal(size=size)
    return random1, rho * random1 + np.sqrt(1 - rho**2) * random2

def monte_carlo(function, num_sim, Z_gen, **kwargs): 
    result = []
    for i in range(num_sim):
        Z = Z_gen[i]
        S = function(**kwargs,Z=Z)
        result.append(S)    
    return result

def monte_carlo_monthly(function, num_sim,len_sim, Z_gen,S0, **kwargs): 
    result = []
    for i in range(num_sim):
        Z_per_sim = Z_gen[i*len_sim:(i+1)*len_sim]
        S_list = []
        S = S0
        S_list.append(S)
        for j in range(len_sim):
            Z = Z_per_sim[j]
            S = function(S,**kwargs,Z=Z)
            S_list.append(S)
        result.append(S_list)
            
    return result

_norm_cdf = stats.norm(0, 1).cdf
_norm_pdf = stats.norm(0, 1).pdf

def _d1(S, K, T, r,q, sigma):
    return (np.log(S / K) + (r-q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

def _d2(S, K, T, r,q, sigma):
    return _d1(S, K, T, r,q, sigma) - sigma * np.sqrt(T)

def bs_put_value(S, K, T, r , q, sigma):
    return np.exp(-r * T) * K * _norm_cdf(-_d2(S, K, T, r,q, sigma)) - S *np.exp(-q * T)* _norm_cdf(-_d1(S, K, T, r,q, sigma))

def forward_value(S,K,T,r,q):
    return S*np.exp(-T*q)-K*np.exp(-T*r)

#----------------------------------Qustion 1.1

Z1,Z2 = correlated_random_values(1000000, 0.8)
SX5E = monte_carlo(equity, 1000000, Z1, St = 4235, r=0.03, q=0.02, vol=0.15 ,delta=5)
AEX = monte_carlo(equity, 1000000, Z2, St = 770, r=0.03, q=0.02, vol=0.15 ,delta=5)
SX5E_forward_value = (np.array(SX5E) - 4235) * np.exp(-5*0.03)
AEX_forward_value = (np.array(AEX) - 770) * np.exp(-5*0.03)
SX5E_forward_mean = np.mean(SX5E_forward_value)
AEX_forward_mean = np.mean(AEX_forward_value)
SX5E_forward_CI = [SX5E_forward_mean-1.96*np.std(SX5E_forward_value)/np.sqrt(len(SX5E_forward_value)),SX5E_forward_mean+1.96*np.std(SX5E_forward_value)/np.sqrt(len(SX5E_forward_value))]
AEX_forward_CI = [AEX_forward_mean-1.96*np.std(AEX_forward_value)/np.sqrt(len(AEX_forward_value)),AEX_forward_mean+1.96*np.std(AEX_forward_value)/np.sqrt(len(AEX_forward_value))]

print(SX5E_forward_mean,SX5E_forward_CI)
print(AEX_forward_mean,AEX_forward_CI)
print(forward_value(4235,4235,5,0.03,0.02))
print(forward_value(770,770,5,0.03,0.02))

SX5E_put_value = np.maximum(3388 - np.array(SX5E), 0) * np.exp(-5 * 0.03)
AEX_put_value = np.maximum(616 - np.array(AEX), 0) * np.exp(-5 * 0.03)
SX5E_put_bs = bs_put_value(4235, 3388, 5, 0.03,0.02, 0.15)
AEX_put_bs = bs_put_value(770, 616, 5, 0.03,0.02, 0.15)

SX5E_put_mean = np.mean(SX5E_put_value)
AEX_put_mean = np.mean(AEX_put_value)
SX5E_put_CI = [SX5E_put_mean-1.96*np.std(SX5E_put_value)/np.sqrt(len(SX5E_put_value)),SX5E_put_mean+1.96*np.std(SX5E_put_value)/np.sqrt(len(SX5E_put_value))]
AEX_put_CI = [AEX_put_mean-1.96*np.std(AEX_put_value)/np.sqrt(len(AEX_put_value)),AEX_put_mean+1.96*np.std(AEX_put_value)/np.sqrt(len(AEX_put_value))]

print(SX5E_put_mean,SX5E_put_CI)
print(AEX_put_mean,AEX_put_CI)
print(SX5E_put_bs)
print(AEX_put_bs)

SX5E_log_return = np.log(np.array(SX5E)/4235) 
AEX_log_return = np.log(np.array(AEX)/4235) 
log_return_corr = np.corrcoef(SX5E_log_return, AEX_log_return)[0][1]

n = len(SX5E_log_return)  
se = 1 / np.sqrt(n - 3)

fisher_transform = np.log((1+log_return_corr)/(1-log_return_corr))/2
ci_lower, ci_upper = fisher_transform - 1.96 * se, fisher_transform + 1.96 * se
ci_lower_corr, ci_upper_corr = (np.exp(2*ci_lower)-1)/(np.exp(2*ci_lower)+1), (np.exp(2*ci_upper)-1)/(np.exp(2*ci_upper)+1)

print(log_return_corr)
print(ci_lower_corr, ci_upper_corr)

# -------------------------------Question 1.2----------------------------------


Z1_mon,Z2_mon = correlated_random_values(60*100000, 0.4)
SX5E_mon = monte_carlo_monthly(equity, 100000,60, Z1_mon, S0 = 4235, r=0.03, q=0.02, vol=0.15, delta=1/12)
AEX_mon = monte_carlo_monthly(equity, 100000,60, Z2_mon, S0 = 770, r=0.03, q=0.02, vol=0.15, delta=1/12)

time_list = np.arange(0,5.0001,1/12)

exposures_with_netting = []
for i in range(len(SX5E_mon)):
    net_exposures = []
    for t in range(len(time_list)):
        T = 5 - time_list[t]
        exposure1 = 10000 * forward_value(SX5E_mon[i][t], 4235, T, 0.03,0.02)
        exposure2 = 55000 * forward_value(AEX_mon[i][t], 770, T, 0.03,0.02)
        exposure3 = 10000 * bs_put_value(SX5E_mon[i][t], 3388, T, 0.03,0.02,0.15)
        exposure4 = 55000 * bs_put_value(AEX_mon[i][t], 616, T, 0.03,0.02, 0.15)
        net_exposure = max(0, exposure1 + exposure2 + exposure3 + exposure4)  # Aggregate and ensure positive
        net_exposures.append(net_exposure)
    print(i)
    exposures_with_netting.append(net_exposures)
    
df = pd.DataFrame(exposures_with_netting, columns=time_list)
file_path = 'D:/Finance/Risk/Case3/exposures_with_netting_low_corr.txt'
df.to_csv(file_path, index=False)





exposures_without_netting = []
for i in range(len(SX5E_mon)):
    total_exposures = []
    for t in range(len(time_list)):
        T = 5 - time_list[t]
        exposure1 = max(0, 10000 * forward_value(SX5E_mon[i][t], 4235, T, 0.03,0.02))
        exposure2 = max(0, 55000 * forward_value(AEX_mon[i][t], 770, T, 0.03,0.02))
        exposure3 = max(0, 10000 * bs_put_value(SX5E_mon[i][t], 3388, T, 0.03,0.02, 0.15))
        exposure4 = max(0, 55000 * bs_put_value(AEX_mon[i][t], 616, T, 0.03,0.02,0.15))
        total_exposure = exposure1 + exposure2 + exposure3 + exposure4
        total_exposures.append(total_exposure)  
    exposures_without_netting.append(total_exposures)
    print(i)
 

df = pd.DataFrame(exposures_without_netting, columns=time_list)
file_path = 'D:/Finance/Risk/Case3/exposure_without_netting.txt'
df.to_csv(file_path, index=False)


exposures_without_netting = []
for i in range(len(SX5E_mon)):
    total_exposures = []
    for t in range(len(time_list)):
        T = 5 - time_list[t]
        exposure1 = max(0, 10000 * forward_value(SX5E_mon[i][t], 4235, T, 0.03,0.02))
        total_exposure = exposure1
        total_exposures.append(total_exposure)  
    exposures_without_netting.append(total_exposures)
    print(i)
 

df = pd.DataFrame(exposures_without_netting, columns=time_list)
file_path = 'D:/Finance/Risk/Case3/exposure1.txt'
df.to_csv(file_path, index=False)

exposures_without_netting = []
for i in range(len(SX5E_mon)):
    total_exposures = []
    for t in range(len(time_list)):
        T = 5 - time_list[t]
        exposure2 = max(0, 55000 * forward_value(AEX_mon[i][t], 770, T, 0.03,0.02))
        total_exposure = exposure2
        total_exposures.append(total_exposure)  
    exposures_without_netting.append(total_exposures)
    print(i)
 

df = pd.DataFrame(exposures_without_netting, columns=time_list)
file_path = 'D:/Finance/Risk/Case3/exposure2.txt'
df.to_csv(file_path, index=False)

exposures_without_netting = []
for i in range(len(SX5E_mon)):
    total_exposures = []
    for t in range(len(time_list)):
        T = 5 - time_list[t]
        exposure3 = max(0, 10000 * bs_put_value(SX5E_mon[i][t], 3388, T, 0.03,0.02, 0.15))
        total_exposure = exposure3
        total_exposures.append(total_exposure)  
    exposures_without_netting.append(total_exposures)
    print(i)
 
df = pd.DataFrame(exposures_without_netting, columns=time_list)
file_path = 'D:/Finance/Risk/Case3/exposure3.txt'
df.to_csv(file_path, index=False)




# simple contract is also done with this code but with different simulations.





