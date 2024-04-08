# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 13:28:28 2024

@author: Jiaqi
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os

os.chdir('D:/Finance/Risk/Case3')
netted = pd.read_csv('exposures_with_netting.txt')
netted.columns = [float(col) for col in netted.columns]


random_index = np.random.randint(0, 99999, size=1000)
for random in random_index:
    plt.plot(netted.columns,netted.iloc[random])
    
plt.xlabel('Time (Year)')
plt.ylabel('Exposure (Euro)')
plt.title('1000 Random Selected Routes for the Netted Exposure')
plt.savefig('random_netted.png')


netted.columns = np.round(netted.columns, decimals=2)
netted_mean = netted.mean()



plt.figure(figsize=(20,6))
plt.plot(netted_mean.index,netted_mean)
plt.xlabel('Time (Year)')
plt.ylabel('Exposure (Euro)')
plt.title('Average Netted Exposure')
plt.savefig('exposure_with_netting.png')
