import numpy as np
from numpy.typing import NDArray

class Solution:

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        # X is (n, m), weights is (m,) -> return (n,) predictions
        # Round to 5 decimal places
        # sum=0
        # for i in range(len(X)):
        #     X[i]=np.dot(X[i],weights[i])
        #     sum+=X[i]
        return np.round(np.dot(X,weights),5)

    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        # Compute mean squared error between predictions and ground truth
        # Round to 5 decimal places
        #Commented out code below is also the correct solution
        # sum=0
        # for i in range(len(model_prediction)):
        #     sum+=(model_prediction[i]-ground_truth[i])**2
            
        # sum=sum/len(model_prediction)
        # return np.round(sum,5).item()

        mse=np.mean((np.array(model_prediction)-np.array(ground_truth))**2)
        return np.round(float(mse),5)
