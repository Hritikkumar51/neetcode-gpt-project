import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        epsilon=1e-7
        sum=0
        for i in range(len(y_true)):
            if y_true[i]==1:
                sum+=np.log(y_pred[i]+epsilon)
            else:
                sum+=np.log(1-y_pred[i]+epsilon)
        
        sum=-1*(sum/len(y_true))
        return np.round(sum,4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        epsilon=1e-7
        sum=0
        for i in range(len(y_true)):
            for j in range(len(y_true[i])):
                if y_true[i][j]==1:
                    sum+=np.log(y_pred[i][j]+epsilon)
                # else:
                    # sum+=np.log(1-y_pred[i][j]+epsilon)
        sum=-1*(sum/len(y_true))
        return round(sum,4)
