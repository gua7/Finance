# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 16:05:31 2024

@author: Jiaqi
"""


import numpy as np
from scipy.stats import norm, skewnorm
import os

os.chdir(r"D:\Finance\Risk\Case1")


def assign_new_rating(mapping,rating, value):
    """
    
    Parameters
    ----------
    mapping : dict
        The mapping of rating before and after shock. In z-scores.
    rating : str
        The base rating.
    value : float
        The merton function results.

    Returns
    -------
    new_rating: str
        The adjusted rating after the shock.

    """

    rating_list = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC', 'D']
    
    for i in range(len(mapping[rating])):
        if value >= mapping[rating][i]:
            return rating_list[i]
        
    
def merton(correlation,common,idiosyncratic):
    """
    
    Return the results of one factor merton model
    
    """
    return np.sqrt(correlation)*common + np.sqrt(1-correlation)*idiosyncratic


def market_value(rating, shock=False):
    """
    
    Return the market value of the bonds before/after shock.

    """
    values = {
        'AAA': {'base': 99.40, 'shocked': 99.50},
        'AA': {'base': 98.39, 'shocked': 98.51},
        'A': {'base': 97.22, 'shocked': 97.53},
        'BBB': {'base': 92.79, 'shocked': 92.77},
        'BB': {'base': 90.11, 'shocked': 90.48},
        'B': {'base': 86.60, 'shocked': 88.25},
        'CCC': {'base': 77.16, 'shocked': 77.88},
        'D': {'base': None, 'shocked': 60.00}  
    }

    if shock:
        return values[rating]['shocked']
    else:
        return values[rating]['base']
    
def portfolio_construction(investment_amount, portfolio_data):
    """
    
    Parameters
    ----------
    investment_amount : float
        The total amount of cash to invest
    portfolio_data : list of dictionary
        The dictionaries containing rating, amount of issuers and percentage to invest per grade
        

    Returns
    -------
    amount_per_issuer_mapping : dictionary
        The amount to invest per issuer per grade.

    """

    amount_per_issuer_mapping = {}
    
    for data in portfolio_data:
        rating = data['rating']
        issuers = data['issuers']
        percentage = data['percentage']
        investment_per_rating = investment_amount * (percentage / 100) / issuers
        market_val = market_value(rating, shock=False)
        amount_per_issuer = investment_per_rating/market_val
        amount_per_issuer_mapping[rating] = [amount_per_issuer for _ in range(issuers)]
   
    return amount_per_issuer_mapping

    
def monte_carlo_one_factor(rating_mapping,issuer_mapping,merton_correlation,iterations):
    """
    
    Parameters
    ----------
    rating_mapping : dictionary
        Rating before/after shock, in z-scores.
    issuer_mapping : dictionary
        The amount of bonds to invest per issuer per rating
    merton_correlation : float
        Correlation in one factor model
    iterations : int
        Numbers of iterations to run monte carlo simulation
        

    Returns
    -------
    portfolio_values: array
        Portfolio values for all monte carlo runs.

    """
    portfolio_values = []
    for i in range(iterations):
        portfolio_value = 0
        common = np.random.normal()
        for rating , weights in issuer_mapping.items():
            idiosyncratics = np.random.normal(size = len(weights))
            for i , weight in enumerate(weights):
                idiosyncratic = idiosyncratics[i]
                merton_value = merton(merton_correlation,common,idiosyncratic)
                new_rating = assign_new_rating(rating_mapping,rating, merton_value)
                market_val = market_value(new_rating,shock=True)
                portfolio_value += market_val*weight
        portfolio_values.append(portfolio_value)
                    
    return np.array(portfolio_values)


def risk_statistics(portfolio_values,investment_amount,threshold, measure_type='VaR'):
    """
    
    Parameters
    ----------
    portfolio_values : list
        Simulated portfolio values
    investment_amount : float
        Initial investment amount
    threshold : float
        Percentage of threshold for VaR/ES
    measure_type : TYPE, optional
        Calculate VaR or ES

    Returns
    -------
    Return VaR or ES 

    """
    
    losses = portfolio_values - investment_amount
    losses_sorted = np.sort(losses)
    
    if measure_type == 'VaR':
       VaR = np.percentile(losses_sorted,threshold)
       return VaR
   
    elif measure_type == 'ES':

       threshold_index = int(np.ceil((threshold/100) * len(losses_sorted)))  
       ES = np.mean(losses_sorted[threshold_index:])
       return ES
   
    else:
       raise ValueError("measure_type must be 'VaR' or 'ES'")
       
       
       
def mainflow(rating_mapping,investment_amount,portfolio_data,merton_correlation,iterations):
    '''
    Combine all the functions before, output the required risk indicators.
    '''
    
    issuer_mapping = portfolio_construction(investment_amount, portfolio_data)       
    portfolio_values = monte_carlo_one_factor(rating_mapping,issuer_mapping,merton_correlation,iterations)
    expected_value = np.mean(portfolio_values)
    VaR90 = risk_statistics(portfolio_values,investment_amount,90, measure_type='VaR')
    VaR995 = risk_statistics(portfolio_values,investment_amount,99.5, measure_type='VaR')
    ES90 = risk_statistics(portfolio_values,investment_amount,90, measure_type='ES')
    ES995 = risk_statistics(portfolio_values,investment_amount,99.5, measure_type='ES')

    return portfolio_values,expected_value,VaR90,VaR995,ES90,ES995



original_probabilities = {
    'AAA': [91.115, 8.179, 0.607, 0.072, 0.024, 0.003, 0.000, 0.000],
    'AA': [0.844, 89.626, 8.954, 0.437, 0.064, 0.036, 0.018, 0.021],
    'A': [0.055, 2.595, 91.138, 5.509, 0.499, 0.107, 0.045, 0.052],
    'BBB': [0.031, 0.147, 4.289, 90.584, 3.898, 0.708, 0.175, 0.168],
    'BB': [0.007, 0.044, 0.446, 6.741, 83.274, 7.667, 0.895, 0.926],
    'B': [0.008, 0.031, 0.150, 0.490, 5.373, 82.531, 7.894, 3.523],
    'CCC': [0.000, 0.015, 0.023, 0.091, 0.388, 7.630, 83.035, 8.818]
}

cumulative_probabilities = {rating: 
    [min(100, sum(original_probabilities[rating][:i+1])) for i in range(len(original_probabilities[rating]))]
    for rating in original_probabilities}
    
rating_mapping = {rating: 
    [norm.ppf(1 - (cum_prob / 100)) for cum_prob in cumulative_probabilities[rating]]
    for rating in cumulative_probabilities}

IG_single_issuer = [
    {'rating': 'AAA', 'issuers': 1, 'percentage': 60},
    {'rating': 'AA', 'issuers': 1, 'percentage': 30},
    {'rating': 'BBB', 'issuers': 1, 'percentage': 10}
]

junk_single_issuer = [
    {'rating': 'BB', 'issuers': 1, 'percentage': 60},
    {'rating': 'B', 'issuers': 1, 'percentage': 35},
    {'rating': 'CCC', 'issuers': 1, 'percentage': 5}
]


IG_hundred_issuer = [
    {'rating': 'AAA', 'issuers': 100, 'percentage': 60},
    {'rating': 'AA', 'issuers': 100, 'percentage': 30},
    {'rating': 'BBB', 'issuers': 100, 'percentage': 10}
]

junk_hundred_issuer = [
    {'rating': 'BB', 'issuers': 100, 'percentage': 60},
    {'rating': 'B', 'issuers': 100, 'percentage': 35},
    {'rating': 'CCC', 'issuers': 100, 'percentage': 5}
]

issuer_data_sets = [
    ('IG_single_issuer', IG_single_issuer),
    ('junk_single_issuer', junk_single_issuer),
    ('IG_hundred_issuer', IG_hundred_issuer),
    ('junk_hundred_issuer', junk_hundred_issuer)
]

correlation_values = [0, 0.33, 0.66, 1]

for issuer_name, issuer_data in issuer_data_sets:
    for correlation in correlation_values:
        portfolio_values, expected_value, VaR90, VaR995, ES90, ES995 = mainflow(rating_mapping,1500*10**6, issuer_data, correlation, 1000000)
        
        base_filename = f"{issuer_name}_correlation_{correlation}"
        portfolio_values_filename = f"{base_filename}_PortfolioValues.txt"
        with open(portfolio_values_filename, "w") as pv_file:
            pv_file.write(f"Issuer: {issuer_name}, Correlation: {correlation}\n")
            pv_file.write("\n".join(map(str, portfolio_values)))
        
        # Save risk metrics to a separate file
        risk_metrics_filename = f"{base_filename}_RiskMetrics.txt"
        with open(risk_metrics_filename, "w") as rm_file:
            rm_file.write(f"Issuer: {issuer_name}, Correlation: {correlation}\n")
            rm_file.write(f"Expected Value: {expected_value}\n")
            rm_file.write(f"VaR90: {VaR90}\n")
            rm_file.write(f"VaR995: {VaR995}\n")
            rm_file.write(f"ES90: {ES90}\n")
            rm_file.write(f"ES995: {ES995}\n")
        
        print(f"Saved portfolio values and risk metrics for {issuer_name} with correlation {correlation}.")
                
        

