import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import LabelEncoder as LE  #for preprocessing and normalization of dataset


def numpy_distance(x,y):                #function use to calculate the distance between two points
    return np.linalg.norm(x-y)

class K_Means_Clustering:
    
    def __init__(self, K=3, max_Iter=100):
        self.K = K
        self.max_Iteration=max_Iter
        self.centroids = []
        self.clusters=[]
        for i in range(self.K):
            self.clusters.append([])
        
        
    def predict(self, X):
        self.pred=X
        self.no_samples, self.no_features = X.shape
        
        rand_sample_index = np.random.choice(self.no_samples, self.K, replace=False)  #creating random sample centroids
        
        for i in rand_sample_index:
            self.centroids.append(self.pred[i])
            
        for i in range(self.max_Iteration):
            
            clusters=[]
            for i in range(self.K):
                clusters.append([])
            for i, sample in enumerate(self.pred):      # creating clusters
                centroid_index = self.centroid(sample, self.centroids)
                clusters[centroid_index].append(i)
                
            self.clusters=clusters
                        
             # updating the centroids untill we found the best match
            centroids_old = self.centroids    
            self.centroids= self.create_centroids(self.clusters)
        
            if self.is_completed(centroids_old, self.centroids):
                break
              
        labels = np.empty(self.no_samples)      #finding the labels

        for index, cluster in enumerate(clusters):
            for sample_index in cluster:
                labels[sample_index] = index
        return labels
            
           
        
    def centroid(self, s, centroids):
        # distance of the current sample to each centroid
        distances = [numpy_distance(s, point) for point in centroids]
        closest_index = np.argmin(distances)
        return closest_index
        
        
    def create_centroids(self, clusters):
    # assign mean value of clusters to centroids
        centroids = np.zeros((self.K, self.no_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.pred[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    def is_completed(self, centroids_old, centroids):
        distances = []
        for i in range(self.K):
            distances.append(numpy_distance(centroids_old[i], centroids[i]))       
        return sum(distances) == 0
    
    def plot(self):
       fig, ax = plt.subplots()

       for i, index in enumerate(self.clusters):
           point = self.pred[index].T
           ax.scatter(*point)
           
       for point in self.centroids:
           ax.scatter(*point, marker="x", linewidth=18)
       
       plt.title("Clusters")
       plt.savefig("Clusters.png")    
       plt.show()
   

#working on dataset
dataSet = pd.read_csv("C:\\Users\\malis\\Desktop\\sem5\\AI_Lab\\Week10\\Iris_DataSet.csv")  # loading the dataset

X_Features = dataSet[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
Y_Feature = dataSet['species'].values

#preprocessing the dataset
Y_Feature=LE.fit_transform(LE,Y_Feature)        #transforming our string data into integers where setosa=0, versicolor=1                                               #virginica=2

#running the K means algorithm
clusters = len(np.unique(Y_Feature))        # getting the number of number of unique classes

k = K_Means_Clustering(clusters, 200)
y_pred = k.predict(X_Features)
k.plot()

from sklearn.metrics import confusion_matrix as CM
import seaborn as sns

confusionMatrix = CM(Y_Feature, y_pred)
print("confusionMatrix is: ")
print(confusionMatrix)


plt.figure()
sns.heatmap(confusionMatrix, annot=True, cmap='Purples')
plt.title("Confusion Matrix")
plt.savefig("Confusion_Matrix.png")
plt.show()



def plot_Corelatoin(dataset):
    plt.figure()
    sns.heatmap(dataset, annot=True, cmap='Purples')
    plt.title("Corelatoin Map")
    plt.savefig("Corelatoin.png")
    plt.show()


def plot_Class():
    plt.figure()
    plt.title("Class Wise Plotting")
    sns.scatterplot(x="sepal_length", y="sepal_width", data=dataSet, hue="species")
    plt.savefig("Class_SL_SW.png")
    plt.show()

    plt.figure()
    plt.title("Class Wise Plotting")
    sns.scatterplot(x="petal_length", y="petal_width", data=dataSet, hue="species")
    plt.savefig("Class_PL_PW.png")
    plt.show()


plot_Corelatoin(dataSet.corr())
plot_Class()
    

