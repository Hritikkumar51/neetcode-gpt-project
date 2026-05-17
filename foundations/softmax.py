import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        sum=0
        maximum_z=0
        for i in range(len(z)):
            maximum_z=max(maximum_z,z[i])
        
        for i in range(len(z)):
            z[i]= np.exp(z[i]-maximum_z)
            sum+=z[i]

        for i in range(len(z)):
            z[i]=np.round(z[i]/sum, 4)
        
        return z
