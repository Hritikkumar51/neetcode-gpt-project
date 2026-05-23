import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        # x: 1D input array
        # w: 1D weight array
        # b: scalar bias
        # y_true: true target value
        #
        # Forward: z = dot(x, w) + b, y_hat = sigmoid(z)
        # Loss: L = 0.5 * (y_hat - y_true)^2
        # Return: (dL_dw rounded to 5 decimals, dL_db rounded to 5 decimals)

        ###########My solution below #########
        # z=0.0
        # for i in range(len(x)):
        #     z+=np.dot(x[i],w[i])
        # z+=b # adding bias

        # y_hat=1/(1+np.exp(-z)) #sigmoid calculation

        # # L=0.5*((y_hat-y_true)**2)

        # grad_L_w=np.round((y_hat-y_true)*y_hat*(1-y_hat)*x,5) ##gradient calculation w.r.t w
        # grad_L_b=np.round(float(y_hat-y_true)*y_hat*(1-y_hat),5) ##gradient calculation w.r.t b but it needs to be in float type

        # return grad_L_w,grad_L_b


        ########## Another simple solution ##########
        z=np.dot(x,w)+b #Forward calculation
        y_hat=1/(1+np.exp(-z)) #sigmoid calculation
        dL_dw=np.round((y_hat-y_true)*y_hat*(1-y_hat)*x,5) #Gradient calculation w.r.t w
        dL_db=np.round(float((y_hat-y_true)*y_hat*(1-y_hat)),5) #Gradient calculation w.r.t b
        return dL_dw,dL_db