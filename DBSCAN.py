# -*- coding: utf-8 -*-
"""
Data Mining Assignment 1

Using a futuristic telescope, ISRO gathered data of nebulas that are several 
lights years away from earth. With some analysis this data is mapped into two 
dimensional data (file galaxy.xlsx provided with this document). 
ISRO's scientific probe is converging to a point that shows the formation of 
five possible star coagulations from these nebulas. As a data scientist you 
are expected to help ISRO to analyze their data further. 

Created on Sun Feb 24 22:05:20 2020

@author: Praveen Saxena	-- 2019ab04190
         Nirmal G. Sivakumar --	2019ab04076
         Sachin Kumar -- 2019ab04207
 
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn import metrics
from numpy import percentile
########################## Read data file #####################################
# Read galaxy xlsx file
galaxyXlsx = pd.ExcelFile('galaxy.xlsx')
X = pd.read_excel(galaxyXlsx)
galaxyDF = pd.DataFrame({'x': X['x'], 'y': X['y']})

########################## KNN to find optimal eps value ######################
# use the standard NearestNeighbors method. Pass value of K = 4 (min pts)
epsValue=.15
minSamples=4
from sklearn.neighbors import NearestNeighbors
neigh = NearestNeighbors(n_neighbors=4, algorithm='auto').fit(galaxyDF)
distances, indices = neigh.kneighbors(galaxyDF)
distances = np.sort(distances,axis = 0)
distances = distances[:,1]
plt.figure(figsize=(10,10)) 
plt.plot(distances)

heading = "Draw the KNN graph to find the optimal value for eps :"
print("\u0332".join(heading))
# Show the grid lines as dark grey lines
plt.grid(b=True, which='major', color='#666666', linestyle='-')
# Show the minor grid lines with very faint and almost transparent grey lines
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.axhline(y=epsValue, color='y', linestyle='-',linewidth=4)
plt.text(900, epsValue, epsValue, fontsize=20, va='center', ha='center', backgroundcolor='w')
plt.show()

# ######################## DBSCAN Algo ########################################
# Execute DBSCAN algorith with value eps = 0.15 and min_samples = 4 
# Compute DBSCAN

dbscan = DBSCAN(eps=epsValue, min_samples=minSamples).fit(galaxyDF)
core_samples_mask = np.zeros_like(dbscan.labels_, dtype=bool)
core_samples_mask[dbscan.core_sample_indices_] = True
labels = dbscan.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

# ######################## Plot DBSCAN ########################################
# Set Plot result for DBSCAN
heading = "Draw DBSCAN plot :"
print("\u0332".join(heading))
plt.figure(figsize=(10,10)) 
plt.title('Estimated number of clusters: %d' % n_clusters_)
# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    labelStr = ''
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]
        labelStr = 'Noise'
    else:
        labelStr = 'Cluster %d'% (k+1)
    class_member_mask = (labels == k)

    xy = galaxyDF[class_member_mask & core_samples_mask]
    plt.plot(xy['x'],xy['y'], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14, label=labelStr)

    xy = galaxyDF[class_member_mask & ~core_samples_mask]
    plt.plot(xy['x'],xy['y'], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6, label=labelStr) 

# Display legend on the plot 
plt.legend(loc='upper center',bbox_to_anchor=(0.5, 1.05), ncol=6, fancybox=True, shadow=True)
plt.show()
########################## Clusters Statistics#################################
# Determining the number of clustes, elements per cluster,
# and outliers (noise points)
heading = "Clusters Statistics elements in density region:"
print("\u0332".join(heading))
print('\tEstimated number of clusters: %d' % n_clusters_)
print("\tSilhouette Coefficient: %0.3f"
      % metrics.silhouette_score(galaxyDF, labels))
for cluster_num in range(n_clusters_):
    print('\tElements in density region %d: %d'% (cluster_num + 1, list(labels).count(cluster_num)))

outliers = list(labels).count(-1)
print('\tTotal outliers/noise: %d' % outliers)
print()
###############################################################################
heading = "Outliers x-y coordinates of the noise points are:"
print("\u0332".join(heading))
print(galaxyDF[labels == -1])
# #############################################################################
# Determining the outliers using quantile method
#find_outlier
def find_outlier(data):
    # calculate interquartile range
    q25, q75 = percentile(data, 25), percentile(data, 75)
    iqr = q75 - q25
    print('\tPercentiles: 25th=%.3f, 75th=%.3f, IQR=%.3f' % (q25, q75, iqr))
    # calculate the outlier cutoff
    cut_off = iqr * 1.5
    lower, upper = q25 - cut_off, q75 + cut_off
    # identify outliers
    outliers = [x for x in data if x < lower or x > upper]
    print('\tIdentified outliers: %d' % len(outliers))
    print('\toutliers : ', outliers)
    # remove outliers
    outliers_removed = [x for x in data if x >= lower and x <= upper]
    print('\tNon-outlier observations: %d' % len(outliers_removed))
    
heading = "\nOutliers analysis for dimension \'x\'"
print("\u0332".join(heading))
find_outlier(X['x'])
heading = "Outliers analysis for dimension \'y\'"
print("\u0332".join(heading))
find_outlier(X['y'])
   