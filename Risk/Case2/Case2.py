# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 07:29:44 2024

@author: Jiaqi
"""
import numpy as np
import pandas as pd

def simple_survival_rate(CDS_rate,year,LGD):
    avg_hazard_rate = CDS_rate/LGD
    survival_rate = np.exp(-avg_hazard_rate/100*year)
    return avg_hazard_rate,survival_rate

def simple_forward_harzard(CDS_A,year_A,CDS_B,year_B,LGD):
    ahr_A,sr_A = simple_survival_rate(CDS_A,year_A,LGD)
    ahr_B,sr_B = simple_survival_rate(CDS_B,year_B,LGD)
    forward_harzard = (ahr_B*year_B - ahr_A*year_A)/(year_B-year_A)
    return forward_harzard

CDS_sheet = [
    {'maturity': 0, 'CDS_rate': 0},
    {'maturity': 1, 'CDS_rate': 1},
    {'maturity': 3, 'CDS_rate': 1.1},
    {'maturity': 5, 'CDS_rate': 1.2},
    {'maturity': 7, 'CDS_rate': 1.2},
    {'maturity': 10, 'CDS_rate': 1.25}
]
df = pd.DataFrame(CDS_sheet)
df_complex = df.copy()

df['ir'] = (1.03**df['maturity']-1)/df['maturity']
df['simple_avg_hazard_rate'],df['survival_rate'] = simple_survival_rate(df['CDS_rate'],df['maturity'],0.4)
df['simple_forward_PD'] = 0
df['simple_forward_hazard_rate'] = 0
for i in range(1,len(df)):
    df['simple_forward_hazard_rate'][i] = simple_forward_harzard(df['CDS_rate'][i-1],df['maturity'][i-1],df['CDS_rate'][i],df['maturity'][i],0.4)
    df['simple_forward_PD'][i] = (df['survival_rate'][i-1] - df['survival_rate'][i])/(df['maturity'][i]-df['maturity'][i-1])    
df.to_excel('result_table1.xlsx', index=False)



def survival_probability_quarter(t, lamb1=None, lamb2=None, lamb3=None, lamb4=None, lamb5=None):
    '''
    

    Parameters
    ----------
    t : float, time in year
        time in year (between 0 to 10 only)
    lamb1 : float, optional
        default intensity between year 0 and 1. The default is None.
    lamb2 : float, optional
        default intensity between year 1 and 3. The default is None.
    lamb3 : float, optional
        default intensity between year 3 and 5. The default is None.
    lamb4 : float, optional
        default intensity between year 5 and 7. The default is None.
    lamb5 : float, optional
        default intensity between year 7 and 10. The default is None.
                    
    Returns
    -------
    survival_prob : float
        Cumulated survival probability

    '''
    lamb_values = [lamb for lamb in (lamb1, lamb2, lamb3, lamb4, lamb5) if lamb is not None]
    intervals = [1, 2, 2, 2, 3]
    accumulated_time = 0
    survival_prob = 1
    for i, duration in enumerate(intervals):
        if t > accumulated_time and t <= accumulated_time + duration:
            survival_prob *= np.exp(-(t - accumulated_time) * lamb_values[i])
            break  
        elif t > accumulated_time + duration:
            survival_prob *= np.exp(-duration * lamb_values[i])
            accumulated_time += duration
        else:
            break  
    return survival_prob


def CDS_quarter(T,CDS_rate,LGD,IR,lamb1=None, lamb2=None, lamb3=None, lamb4=None, lamb5=None):
    time_list =  np.arange(0, T + 0.25, 0.25)
    premium = 0
    accured = 0
    protection = 0
    IR = IR/4
    
    for i in range(1,len(time_list)):
        premium += (np.exp(-IR * time_list[i]) * (time_list[i] - time_list[i-1]) *
                    survival_probability_quarter(time_list[i], lamb1, lamb2, lamb3, lamb4, lamb5))
        
        accured += (np.exp(-IR*(time_list[i]+time_list[i-1])/2) *
                    (survival_probability_quarter(time_list[i-1], lamb1, lamb2, lamb3, lamb4, lamb5)-
                    survival_probability_quarter(time_list[i], lamb1, lamb2, lamb3, lamb4, lamb5)) *
                    (time_list[i]-time_list[i-1])/2)
        
        protection += (np.exp(-IR*(time_list[i]+time_list[i-1])/2) * LGD *
                      (survival_probability_quarter(time_list[i-1], lamb1, lamb2, lamb3, lamb4, lamb5)-
                        survival_probability_quarter(time_list[i], lamb1, lamb2, lamb3, lamb4, lamb5)))
        
    CDS_price = CDS_rate * (premium + accured) - protection
    return CDS_price


def find_CDS_solution(T,CDS_rate,LGD,IR,maximum_iteration,*lambs):
    x_low = 0
    x_high = 1
    solution_low = CDS_quarter(T,CDS_rate,LGD,IR,*lambs,x_low)

    for _ in range(maximum_iteration):
        solution_med = CDS_quarter(T,CDS_rate,LGD,IR,*lambs,(x_high+x_low)/2)
        if np.abs(solution_med) < 0.000000000000001:
            return (x_high+x_low)/2
        
        elif np.sign(solution_low) == np.sign(solution_med):
            x_low = (x_high+x_low)/2
        
        else:
            x_high = (x_high+x_low)/2  
    return (x_high+x_low)/2
    
lamb1 = find_CDS_solution(1, 0.01, 0.4, 0.03,100000000)  
lamb2 = find_CDS_solution(3, 0.011, 0.4, 0.03,100000000,lamb1)   
lamb3 = find_CDS_solution(5, 0.012, 0.4, 0.03,100000000,lamb1,lamb2)
lamb4 = find_CDS_solution(7, 0.012, 0.4, 0.03,100000000,lamb1,lamb2,lamb3)
lamb5 = find_CDS_solution(10, 0.0125, 0.4, 0.03,100000000,lamb1,lamb2,lamb3,lamb4)
lambdas = np.array([0,lamb1,lamb2,lamb3,lamb4,lamb5])
years = [0,1,3,5,7,10]
interval = np.array([0,1,2,2,2,3])
average_hazards = np.zeros((len(lambdas)-1))
for i in range(1,len(lambdas)):
    average_hazards[i-1] = sum(lambdas[:(i+1)]*interval[:(i+1)])/(years[i])
print(lambdas[1:])
print(average_hazards)

survival_proba1 = survival_probability_quarter(1, lamb1)
survival_proba2 = survival_probability_quarter(3, lamb1, lamb2)
survival_proba3 = survival_probability_quarter(5, lamb1, lamb2, lamb3)
survival_proba4 = survival_probability_quarter(7, lamb1, lamb2, lamb3, lamb4)
survival_proba5 = survival_probability_quarter(10, lamb1, lamb2, lamb3, lamb4, lamb5)
survival = np.array([survival_proba1,survival_proba2,survival_proba3,survival_proba4,survival_proba5])
total_default = 1 - survival
avg_default = np.zeros((len(total_default)))
avg_default[0] = total_default[0]
for i in range(1,len(total_default)):
    avg_default[i] = (total_default[i] - total_default[i-1])/interval[i+1]
print(avg_default)
    