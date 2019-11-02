# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 20:20:32 2019

@author: tinku
"""

import numpy as np
import random
import pandas as pd
from matplotlib import pyplot as plt
from copy import deepcopy


plt.rcParams['figure.figsize'] = (16, 9)    #Makes a image of 16 inches x 9 inches
plt.style.use('ggplot')   # a popular plotting package

# Reading Excel (.xlsx) file
column_names=['X','Y']
file='dataset.xlsx'
data= pd.read_excel(file,names=column_names)

#Fetching X and Y coordinates values
x1 = data['X'].values
y1 = data['Y'].values
X=np.array(list(zip(x1,y1)))

# Function: Euclidean distance
# ---------------------
# Euclidean distance is a function that takes in a and b
# and returns the euclidean distance between 'a' and each element of 'b'
def euclidean_distance(a, b,ax=1):
    distances=np.linalg.norm(a-b,axis=ax);             
    return distances;


# Function: K Means
# -------------
# K-Means is an algorithm that takes in a dataset and a constant
# k and returns k centroids (which define clusters of data in the
# dataset which are similar to one another).
def kmeans(dataset,k,centroids):
    old_centroids=np.zeros(centroids.shape)
    diff=should_stop_iterations(old_centroids,centroids)
    while (diff !=0):
        old_centroids=deepcopy(centroids);
        labels=getLabels(dataset,centroids,k)
        centroids=getCentroids(dataset,labels,k,centroids)        
        diff=should_stop_iterations(old_centroids,centroids)      
    return centroids;
    

# Function: Get Labels
# -------------
# Returns a label for each piece of data in the dataset. 
def getLabels(dataset,centroids,k):
    labels=[0]*len(dataset)
    for i in range(len(dataset)):
        distances=euclidean_distance(dataset[i],centroids)
        labels[i]=int(np.argmin(distances))
    return labels;


# Function: Get Centroids
# -------------
# Returns k centroids, each of dimension n=2.
# Each centroid is the geometric mean of the points that 
# have that centroid's label.
def getCentroids(dataset,labels,k,centroids):
    for i in range(k):
        cluster_points=[dataset[j] for j in range(len(dataset)) if labels[j]==i]
        #print(cluster_points)
        centroids[i]=np.mean(cluster_points,axis=0)
    return centroids;


# Function: Should_Stop_Iterations
# -------------
# Returns True or False if k-means is done. 
# K-means terminates if the centroids stop changing.
def should_stop_iterations(old_centroids,centroids):
    return euclidean_distance(centroids,old_centroids,None);


# Function: Sum_of_squared_error
# -------------------
# Sum_of_squared_error takes in dataset and centroids and the labels assigned
# to each datapoint and return the sum_of_squared_error of that classification
# by computing the sum of square of the euclidean distance of 
# each data point to the centroid assigned to it
def sum_of_squared_error(dataset,labels,centroids):
    error=0.0;
    for i in range(len(centroids)):
        points=[dataset[j] for j in range(len(dataset)) if labels[j]==i]
        error+=np.sum(np.square(euclidean_distance(centroids[i],points)))
    return error;

 
# Function:visualise data
# ------------------
# Visualise_data is a function to visualise the dataset into the clusters
# resulted by applying kmeans algorithm
def visualise_data(centroids,labels,dataset,fig_num):
    fignames=['clusters_k=3 with initialisation with given centers','clusters_k=1','clusters_k=2','clusters_k=4','clusters_k=5','clusters_k=6','clusters_k=7','clusters_k=3 with random initialisation of centroids']
    colors = ['g' ,'y' ,'r' ,'b' ,'c' ,'m','brown']
    plt.figure()
    for i in range(len(centroids)):
        points=[j for j in range(len(dataset)) if labels[j]==i]
        plt.scatter(x1[points],y1[points],c=colors[i],s=10)
    c_x=[item[0] for item in centroids]
    c_y=[item[1] for item in centroids]    
    plt.scatter(c_x,c_y,c='black',s=200,marker='*')
    plt.xlabel('X--->')
    plt.ylabel('Y--->')
    plt.title(fignames[fig_num])
    plt.savefig(fignames[fig_num])
    plt.show()
    plt.close()
    return;



C_xvalue=[3 ,6 ,8]
C_yvalue=[3 ,2 ,5]
C = np.array(list(zip(C_xvalue, C_yvalue)),dtype=np.float32)
centroids=kmeans(X,3,C)
labels=getLabels(X,centroids,3)
error=sum_of_squared_error(X,labels,centroids)
visualise_data(centroids,labels,X,0)
print(' For k=3 with given initial cluster centers :')
print('Centroids :','\n',centroids)
print('Error :',error)



k_values=[1,2,4,5,6,7,3]
errors=np.zeros(len(k_values))
for i in range(len(k_values)):
    idx=np.random.randint(0,len(X),k_values[i])    
    C=np.array(X[idx])
    centroids=kmeans(X,k_values[i],C)
    labels=getLabels(X,centroids,k_values[i])
    errors[i]=sum_of_squared_error(X,labels,centroids)
    visualise_data(centroids,labels,X,i+1)
    print('For k =',k_values[i],'with random initial centers :')
    print('Centroids :','\n',centroids)
    print('Error :',errors[i])
    
 

errors[len(k_values)-1]=error;
k_values_new,errors_new=zip(*sorted(zip(k_values,errors)));
plt.plot(k_values_new,errors_new,'bx-')
plt.xlabel('k--->')
plt.ylabel('Error---->')
plt.title('Error vs k plot showing optimal k')
plt.savefig('k-elbow.png')


