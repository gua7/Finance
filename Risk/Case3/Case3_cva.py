# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 15:52:02 2024

@author: Jiaqi
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os

os.chdir('D:/Finance/Risk/Case3')

def calculatePD(fhra,ta,fhrb,tb):
    PD = np.exp(-ta*fhra)-np.exp(-tb*fhrb)
    return PD

def calculateCVA(simulated_df,PD_list,LGD,IR):
    CVA_list = []
    for year in simulated_df.columns:
        year_float = float(year)
        if year_float<=1:
            PD = PD_list[0]/12
        elif year_float<=3:
            PD = PD_list[1]/24
        else:
            PD = PD_list[2]/24
            
        
        CVA = LGD*PD*np.exp(-IR*year_float)*simulated_df[year].mean()
        CVA_list.append(CVA)
    return CVA_list

def calculateCVAcolletral(simulated_df,PD_list,LGD,IR,collateral_period):
    CVA_list = []
    colletral_amount = 0
    for i , year in enumerate(simulated_df.columns):
        year_float = float(year)
        if year_float<=1:
            PD = PD_list[0]/12
        elif year_float<=3:
            PD = PD_list[1]/24
        else:
            PD = PD_list[2]/24
        
        if collateral_period == 0:
            colletral_amount = 0
        elif i % collateral_period == 0:
            colletral_amount = simulated_df[year].mean()
        
        CVA = LGD*PD*np.exp(-IR*year_float)*(simulated_df[year]-colletral_amount).mean()  

        CVA_list.append(CVA)
    return CVA_list

def calculateCVAim(simulated_df,PD_list,LGD,IR,IM):
    CVA_list = []
    for year in simulated_df.columns:
        year_float = float(year)
        if year_float<=1:
            PD = PD_list[0]/12
        elif year_float<=3:
            PD = PD_list[1]/24
        else:
            PD = PD_list[2]/24
            
        
        CVA = LGD*PD*np.exp(-IR*year_float)*np.maximum(simulated_df[year] - IM, 0).mean()
        CVA_list.append(CVA)
    return CVA_list

def calculateCDS(LGD,forward_hazard_list,time_interval):
    average_hazard_rate = np.sum(forward_hazard_list*time_interval)/np.sum(time_interval)
    CDS = LGD * average_hazard_rate
    return CDS



PD01 = calculatePD(0,0,0.02,1)
PD01_increase = calculatePD(0,0,0.021,1)
PD13 = calculatePD(0.02, 1, 0.0215, 3)
PD13_increase = calculatePD(0.02, 1, 0.0225, 3)
PD35 = calculatePD(0.0215, 3, 0.022, 5)
PD35_increase = calculatePD(0.0215, 3, 0.023, 5)
PD_list = [PD01,PD13,PD35]
PD_list01 = [PD01_increase,PD13,PD35]
PD_list13 = [PD01,PD13_increase,PD35]
PD_list35 = [PD01,PD13,PD35_increase]

e1 = pd.read_csv('exposure1.txt')
e2 = pd.read_csv('exposure2.txt')
e3 = pd.read_csv('exposure3.txt')
e4 = pd.read_csv('exposure4.txt')
e_nonet = pd.read_csv('exposure_without_netting.txt')
e_netted = pd.read_csv('exposures_with_netting.txt')
e_high_vol = pd.read_csv('exposures_with_netting_high_vol.txt')
e_low_corr = pd.read_csv('exposures_with_netting_low_corr.txt')

CVA_e1 = calculateCVA(e1, PD_list, 0.4, 0.03)
CVA_e2 = calculateCVA(e2, PD_list, 0.4, 0.03)
CVA_e3 = calculateCVA(e3, PD_list, 0.4, 0.03)
CVA_e4 = calculateCVA(e4, PD_list, 0.4, 0.03)
CVA_e5 = calculateCVA(e_nonet, PD_list, 0.4, 0.03)

CVA_e6 = calculateCVA(e_netted, PD_list, 0.4, 0.03)
CVA_e6_01 = calculateCVA(e_netted, PD_list01, 0.4, 0.03)
CVA_e6_13 = calculateCVA(e_netted, PD_list13, 0.4, 0.03)
CVA_e6_35 = calculateCVA(e_netted, PD_list35, 0.4, 0.03)

CVA_e7 = calculateCVA(e_high_vol, PD_list, 0.4, 0.03)
CVA_e8 = calculateCVA(e_low_corr, PD_list, 0.4, 0.03)

print(np.sum(CVA_e1),np.sum(CVA_e2),np.sum(CVA_e3),np.sum(CVA_e4),np.sum(CVA_e5),np.sum(CVA_e6))
print(np.sum(CVA_e7),np.sum(CVA_e8))
print(np.sum(CVA_e6_01)-np.sum(CVA_e6),np.sum(CVA_e6_13)-np.sum(CVA_e6),np.sum(CVA_e6_35)-np.sum(CVA_e6))

collateral_period_list = [0,1,2,3,12,24,36,48,60]

for collateral_period in collateral_period_list:
    CVA_list = calculateCVAcolletral(e_netted,PD_list,0.4,0.03,collateral_period)
    plt.plot([float(x) for x in e_netted.columns],CVA_list,label=f'{collateral_period} months')
plt.legend()
plt.xlabel('Time (Year)')
plt.ylabel('CVA charges')
plt.title('CVA charges with different collateral period')
plt.savefig('CVA_collateral.png')


im_list = [10**6,10**7,10**8]
for im in im_list:
    CVA_list = calculateCVAim(e_netted,PD_list,0.4,0.03,im)
    plt.plot([float(x) for x in e_netted.columns],CVA_list,label=f'{im} initial margin')
plt.legend()
plt.xlabel('Time (Year)')
plt.ylabel('CVA charges')
plt.title('CVA charges with different initial margins')
plt.savefig('CVA_im.png')


CDS01 = calculateCDS(0.4,np.array([0.02]),np.array([1]))
CDS01_increase01 = calculateCDS(0.4,np.array([0.021]),np.array([1]))
CDS13 = calculateCDS(0.4,np.array([0.02,0.0215]),np.array([1,2]))
CDS13_increase01 = calculateCDS(0.4,np.array([0.021,0.0215]),np.array([1,2]))
CDS13_increase13 = calculateCDS(0.4,np.array([0.02,0.0225]),np.array([1,2]))
CDS35 = calculateCDS(0.4,np.array([0.02,0.0215,0.022]),np.array([1,2,2]))
CDS35_increase01 = calculateCDS(0.4,np.array([0.021,0.0215,0.022]),np.array([1,2,2]))
CDS35_increase13 = calculateCDS(0.4,np.array([0.02,0.0225,0.022]),np.array([1,2,2]))
CDS35_increase35 = calculateCDS(0.4,np.array([0.02,0.0215,0.023]),np.array([1,2,2]))
print(CDS01,CDS13,CDS35)
print(CDS01_increase01-CDS01,CDS13_increase01-CDS13,CDS35_increase01-CDS35)
print(CDS13_increase13-CDS13,CDS35_increase13-CDS35)
print(CDS35_increase35-CDS35)

