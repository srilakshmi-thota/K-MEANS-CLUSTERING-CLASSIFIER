# K-MEANS-CLUSTERING-CLASSIFIER
K-means Clustering algorithm is used to classify,experimenting with different values of K to find the elbow point in the plot error vs K

__Libraries used:__ \
->numpy  for using arrays and calculating sum and mean of array\
->pandas  for reading the excel file

__Inputs:__\
->dataset.xlsx : excel sheet containing the dataset that need to be classified into clusters.

__Outputs:__\
->For K=3 with given intial centroids:Plot displaying the classfied clusters of the dataset 

->For K=1 : Plot displaying the classfied clusters of the dataset by taking random  initial centroids\
->For K=2:Plot displaying the classfied clusters of the dataset by taking random initial centroids\
->For K=4:Plot displaying the classfied clusters of the dataset by taking random initial centroids\
->For K=5:Plot displaying the classfied clusters of the dataset by taking random initial centroids\
->For K=6:Plot displaying the classfied clusters of the dataset by taking random initial centroids\
->For K=7:Plot displaying the classfied clusters of the dataset by taking random initial centroids

->For K=3:Plot displaying the classfied clusters of the dataset by taking random initial centroids\
->Error vs K plot

__User defined functions:__\
__1.euclidean_distance__\
->Inputs:a,b\
->Outputs:distances\
->Euclidean distance is a function that takes in a and b and returns the distances array in which d[i][j] is the euclidean distances between a[i] and b[j]

__2.kmeans__\
->Inputs:dataset,k,centroids\
->Outputs:centroids\
->k-means is an algorithm that takes in dataset and a constant k denoting the number of clusters and the intial centroids and returns the final centroids which define the clusters of data in the dataset which are similar to one another.

__3.getLabels__\
->Inputs:dataset,centroids,k\
->Outputs:labels\
->Returns a labels array containning the cluster label to which each datapoint in the dataset belong to by evaluating the euclidean distance of the datapoint to the centroids and assigning the nearest centroid label to the datapoint.

__4.getCentroids__\
->Inputs:dataset,labels,k,centroids\
->Outputs:centroids\
->Returns updated k centroids each of dimension 2.Each centroid is the geometric mean of the points that have that centroid's label.

__5.should_stop_iterations__\
->Inputs:old_centroids,centroids\
->Outputs:sum of the distance btwn the old and new centroids\
->Returns 0 if k-means is done by checking the termination condition.K-means terminates if the centroids stop changing.Else returns the sum of the euclidean distance btwn the corresponding old and new centroids.

__6.sum_of_squared_error__\
->Inputs:dataset,labels,centroids\
->Outputs:error\
->Computes the error by taking the sum of the square of the euclidean distance of the dataset point to the corresponding  centroid assigned to it.

__7.visualise_data__\
->Inputs:centroids,labels,dataset,fig_num\
->Outputs:plot\
->Visualise the dataset into the clusters formed as a resultant of appyling k-means clustering to the dataset.

