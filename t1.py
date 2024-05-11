from array import *
import numpy as np
##from sklearn.datasets import make_regression
class LR():
  
  def __init__(self,N):
    self.theta = np.zeros(N)


  def train(self,X, y):

    np_x=np.array(X)
    np_y=np.array(y)
    X_transpose = np_x.T
    X_transpose_X = np.dot(X_transpose, np_x)
    X_transpose_y = np.dot(X_transpose, np_y)

    try:
        self.theta =np.dot(np.linalg.inv(X_transpose_X),X_transpose_y)
        
    except np.linalg.LinAlgError:
        return None
   

  def predict(self, X):
    
    predictions = np.dot(X, self.theta)
  
    return predictions
  

  
p= LR(2)
p.train([[1,1],[2,3]],[4,8])
X_test = np.array([7, 8])
predictions = p.predict(X_test)
print("Predictions:", predictions)


  
    