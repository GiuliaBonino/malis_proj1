
from scipy.spatial import distance_matrix
import numpy as np

import math
class KNN:
    '''
    k nearest neighboors algorithm class
    __init__() initialize the model
    train() trains the model
    predict() predict the class for a new point
    '''

    def __init__(self, k,p):
        '''
        INPUT :
        - k : is a natural number bigger than 0 
        '''

        if k <= 0:
            raise Exception("Sorry, no numbers below or equal to zero. Start again!")
            
        # empty initialization of X and y
        self.X = []
        self.y = []
        # k is the parameter of the algorithm representing the number of neighborhoods
        self.k = k
        self.p=p
        
    def train(self,X,y):
        '''
        INPUT :
        - X : is a 2D NxD numpy array containing the coordinates of points
        - y : is a 1D Nx1 numpy array containing the labels for the corrisponding row of X
        '''    
        self.X=X
        self.y=y
        return    
       
    def predict(self,X_new):
        '''
        INPUT :
        - X_new : is a MxD numpy array containing the coordinates of new points whose label has to be predicted
        A
        OUTPUT :
        - y_hat : is a Mx1 numpy array containing the predicted labels for the X_new points
        ''' 
        dist_matrix=self.minkowski_dist(X_new)
        indeces=np.argsort(dist_matrix)[:,:self.k] #we get the indices of the k nearest points
        classes=self.y[indeces] #we get the classes of the k nearest points
        classes=np.reshape(classes, (classes.shape[0], classes.shape[1])).astype(int)
        y_hat=[]
    
        for row in classes:
            
            counts=np.bincount(row) #maybe add weights here, maybe also do it without the loop
           #print(f"counts{counts}")
            maxclass=counts.argmax()
            y_hat.append(maxclass)

        return y_hat
    
    def minkowski_dist(self,X_new):
        '''
        INPUT : 
        - X_new : is a MxD numpy array containing the coordinates of points for which the distance to the training set X will be estimated
        - p : parameter of the Minkowski distance
        
        OUTPUT :
        - dst : is an MxN numpy array containing the distance of each point in X_new to X
        '''
        first=True

        for x in X_new:
            
            dist=np.sum(abs(x-self.X)**self.p, axis=1)**(1/self.p)

            
            if first:
                dist_matrix=dist
                first=False
                
            else:
                dist_matrix=np.vstack((dist_matrix,dist))
            

        return dist_matrix
    
def main():
    import pandas as pd
    data=pd.read_csv(r"C:\Users\Giulia\Documents\EURECOM\MALIS\proj1\data\training.csv",sep=',') 
    print(data.head(10))
    training_data=np.array(data[['X1','X2']])
    training_y=np.array(data[['y']])
    validation=pd.read_csv(r"C:\Users\Giulia\Documents\EURECOM\MALIS\proj1\data\validation.csv")
    val_data=np.array(validation[['X1','X2']])
    val_y=np.array(validation[['y']])
    errors=[]
    for p in range(1,4):
        print(f"p is {p}")
        for k in range(1,10):
            print(f"k is {k}")
            knn=KNN(k,p)
            
            knn.train(training_data, training_y)
        

            y_hat=np.array(knn.predict(val_data)).astype(int)
            val_y=np.reshape(val_y,(val_y.shape[0]))
            print(y_hat.shape)
            print(val_y.shape)
            error=np.sum(abs(y_hat-val_y))
            errors.append(error)

            print(f"the error is: {error} out of {len(y_hat)}")
    print(min(errors))

    from sklearn.neighbors import NearestNeighbors
    neigh = NearestNeighbors(n_neighbors=2)
    neigh.fit(training_data, training_y)
    y_sk=neigh.predict(val_data)
    error=np.sum(abs(y_sk-val_y))

def main2():
    import pandas as pd
    data=pd.read_csv(r"C:\Users\Giulia\Documents\EURECOM\MALIS\proj1\data\training.csv",sep=',') 
    print(data.head(10))
    training_data=np.array(data[['X1','X2']])
    training_y=np.array(data[['y']])
    validation=pd.read_csv(r"C:\Users\Giulia\Documents\EURECOM\MALIS\proj1\data\validation.csv")
    val_data=np.array(validation[['X1','X2']])
    val_y=np.array(validation[['y']])
    
   
    knn=KNN(2,2)
    
    knn.train(training_data, training_y)


    y_hat=np.array(knn.predict(val_data)).astype(int)
    val_y=np.reshape(val_y,(val_y.shape[0]))
    print(y_hat.shape)
    print(val_y.shape)
    error=np.sum(abs(y_hat-val_y))
    

    print(f"the error is: {error} out of {len(y_hat)}")
    print(error)

    from sklearn.neighbors import KNeighborsClassifier
    neigh =KNeighborsClassifier(n_neighbors=2)
    neigh.fit(training_data, training_y)
    y_sk=neigh.predict(val_data)
    error=np.sum(abs(y_sk-val_y))
    print(error)

main2()
