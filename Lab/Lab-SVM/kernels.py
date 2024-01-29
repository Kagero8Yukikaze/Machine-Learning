import numpy as np


class Base_kernel():
    
    def __init__(self):
        pass
    
    def __call__(self, x1, x2):
        """
        Linear kernel function.
        
        Arguments:
            x1: shape (n1, d)
            x2: shape (n2, d)
            
        Returns:
            y : shape (n1, n2), where y[i, j] = kernel(x1[i], x2[j])
        """
        pass


class Linear_kernel(Base_kernel):
    
    def __init__(self):
        super().__init__()
    
    def __call__(self, x1, x2):
        # TODO: Implement the linear kernel function

        y = np.dot(x1, x2.T)
        
        return y
    
    
class Polynomial_kernel(Base_kernel):
        
    def __init__(self, degree, c):
        super().__init__()
        self.degree = degree
        self.c = c
        
    def __call__(self, x1, x2):
        # TODO: Implement the polynomial kernel function

        y = (np.dot(x1, x2.T) + self.c) ** self.degree
        
        return y

class RBF_kernel(Base_kernel):
    
    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma 
        
        
    def __call__(self, x1, x2):
        # TODO: Implement the RBF kernel function

        # x1 is (n1,d) and x2 is (n2,d)

        x12d = np.atleast_2d(x1) # <- OMG! x1 might not be (n1,d)
        x22d = np.atleast_2d(x2)
        n1, d = x12d.shape
        n2, _ = x22d.shape
        
        x1_reshaped = x12d.reshape((n1, 1, d))
        x2_reshaped = x22d.reshape((1, n2, d))
        
        # x1_reshaped-x2_reshaped is a (n1,n2,d) matrix and then turn into a (n1,n2) matrix
        norm_squared = np.sum((x1_reshaped - x2_reshaped) ** 2, axis=2)
        y = np.exp(-norm_squared / (2 * self.sigma**2))

        return y