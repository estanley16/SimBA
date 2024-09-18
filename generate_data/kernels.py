import numpy as np
from abc import ABC, abstractmethod
 
'''
author: matthias wilms 
'''
class KernelBase(ABC):
 
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def apply(self, x, y):
        pass

class LinearKernel(KernelBase):
    def apply(self, x, y):
        return np.dot(x,y)

class CovKernel(KernelBase):
    def __init__(self, factor):
        super().__init__()
        self.factor=factor

    def apply(self, x, y):
        return np.matrix(self.factor*x@y.T)

class OnesKernel(KernelBase):
    def __init__(self):
        super().__init__()

    def apply(self,x,y=None):
        if y is None:
            return np.ones(x.shape)
        else:
            raise ValueError('OnesKernel: Cannot handle two inputs!')

class GaussianKernel(KernelBase):
    def __init__(self, sigma):
        super().__init__()
        self.sigma=sigma

    def apply(self,x,y=None):
        if y is None:
            return np.exp(-x**2/(self.sigma**2))
        else:
            x_r_sq=np.sum(np.square(x),axis=1)/2
            y_r_sq=np.sum(np.square(y),axis=1)/2
            return np.exp(((x@y.T)-x_r_sq-y_r_sq.T)/(self.sigma**2))

class GaussianKernelNoSquare(KernelBase):
    def __init__(self, sigma):
        super().__init__()
        self.sigma=sigma**2

    def apply(self,x,y=None):
        if y is None:
            return np.exp(-x/(self.sigma))
        else:
            raise ValueError('GaussianKernelNoSquare: Cannot handle two inputs!')

class ExponentialKernel(KernelBase):
    def __init__(self, lambd, q):
        super().__init__()
        self.lambd=lambd
        self.q=q

    def apply(self,x,y=None):
        if y is None:
            return np.exp((-x**self.q)*self.lambd)
        else:
            raise ValueError('ExponentialKernel: Cannot handle two inputs (NYI)!')

class LaplacianKernel(KernelBase):
    def __init__(self, sigma):
        super().__init__()
        self.sigma=sigma

    def apply(self,x,y=None):
        if y is None:
            return np.exp(-np.abs(x)/self.sigma)
        else:
            raise ValueError('LaplacingKernel: Cannot handle two inputs (NYI)!')

#approximation of a heaviside function using Gaussians; However, the kernel is not psd because of the constant used in the denominatior!
class GaussianHeavidiseKernel(KernelBase):
    def __init__(self, sigma):
        super().__init__()
        self.sigma=sigma
        self.gaussian=GaussianKernel(self.sigma)

    def apply(self,x,y=None):
        if y is None:
            return self.gaussian.apply(x)/(self.gaussian.apply(x)+self.gaussian.apply(self.sigma))
        else:
            raise ValueError('GaussianHeavidiseKernel: Cannot handle two inputs!')

class HeavisideKernel(KernelBase):
    def __init__(self, tau):
        super().__init__()
        self.tau=tau

    def apply(self,x,y=None):
        if y is None:
            x_copy=np.copy(x)
            x_copy[x>self.tau]=0
            x_copy[x<=self.tau]=1
            return x_copy
        else:
            raise ValueError('GaussianHeavidiseKernel: Cannot handle two inputs!')


class WendlandKernel3DC0(KernelBase):
    #see Holger Wendland's PhD thesis http://webdoc.sub.gwdg.de/ebook/e/2000/mathe-goe/Hwend2.pdf
    def __init__(self, support_limit):
        super().__init__()
        self.support_limit=support_limit

    def apply(self,x,y=None):
        if y is None:
            return np.maximum(1-(x/self.support_limit),0)**2
        else:
            raise ValueError('WendlandKernel3DC0: only operates on distances!')

class WendlandKernel3DC2(KernelBase):
    #see Holger Wendland's PhD thesis http://webdoc.sub.gwdg.de/ebook/e/2000/mathe-goe/Hwend2.pdf
    def __init__(self, support_limit):
        super().__init__()
        self.support_limit=support_limit

    def apply(self,x,y=None):
        if y is None:
            r=x/self.support_limit
            return np.maximum(1-r,0)**4*(4*r+1)
        else:
            raise ValueError('WendlandKernel3DC2: only operates on distances!')

class WendlandKernel3DC4(KernelBase):
    #see Holger Wendland's PhD thesis http://webdoc.sub.gwdg.de/ebook/e/2000/mathe-goe/Hwend2.pdf
    def __init__(self, support_limit):
        super().__init__()
        self.support_limit=support_limit

    def apply(self,x,y=None):
        if y is None:
            r=x/self.support_limit
            return np.maximum(1-r,0)**6*(35*r**2+18*r+3)/3 #--> normalized with respect to r=0
        else:
            raise ValueError('WendlandKernel3DC4: only operates on distances!')

class WendlandKernel3DC6(KernelBase):
    #see Holger Wendland's PhD thesis http://webdoc.sub.gwdg.de/ebook/e/2000/mathe-goe/Hwend2.pdf
    def __init__(self, support_limit):
        super().__init__()
        self.support_limit=support_limit

    def apply(self,x,y=None):
        if y is None:
            r=x/self.support_limit
            return np.maximum(1-r,0)**8*(32*r**3+25*r**2+8*r+1)
        else:
            raise ValueError('WendlandKernel3DC6: only operates on distances!')
        





    


