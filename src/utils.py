
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la


########## Check the type of a critical point #####################
### We will compute eigenvalues of the Hessian of the function f at a given point z, for example the point x_n in our GD process. This gives us an indication of the type (minimum, maximum) of the limit point (when exists) of the sequence {x_j}.
def CheckCriticalType(fHessian, z):
    print("The point is:", z)
    tmp, w = fHessian(z)
    print("The Hessian matrix is", tmp)
    print("The eigenvalues of Hessian matrix are", w)
    # if (w[0]<0) or (w[1]<0):
    #    print("The point is a generalised saddle point")
    # elif (w[0]==0) or (w[1]==0):
    #    print("The point is degenerate")
    # else:
    #    print("The point is an isolated - nondegenerate local minimum")
    return


############ Bound for learning rates in Unbounded Backtracking GD
def UnboundedLR(t, s, delta0):
    if (t == 0):
        tmp = s
    elif (t < 1):
        tmp = delta0 / math.sqrt(t)
    else:
        tmp = delta0

    return tmp




######### Square of L2 norm of a vector
def L2Norm2(z,dimN):
    if dimN == 1:
        tmp = z * z
    else:
        w = z.flatten()
        tmp = sum(w * w)
    return tmp

#########  Check Armijo's condition
def ArmijoCondition2(f, fDer, z, delta, alpha,atol, dimN):
    g = fDer(z)
    check = f(z - delta * fDer(z)) - f(z) + alpha * delta * L2Norm2(g,dimN)
    tmp = (check < atol)
    return tmp


 
######## Orthogonal decomposition
## For an invertible symmetric matrix A, and a vector x, we decompose x into eigenspaces of positive and negative eigenvalues of A

def NegativeOrthogonalDecomposition(A,x,dimN):
    if dimN>1:
    
        evals, evecs =la.eig(A)
        evals=evals.real
        tmp =0
        for i in range(dimN):
            v=evecs[:,i]
            xw=np.dot(x,v)*v
            if evals[i]<0:
                tmp=tmp-xw
            else:
                tmp=tmp+xw
    else:
        tmp=x
        if A<0:
            tmp=-x
            
    
    return tmp
    
################ Cut-off function
## This is a functions which is not to big. We also want it to be 0 at 0, and no where else. It is used in New Q-Newton's method

def cutoff(t):
    alp=0.51
    #if t>1:
    #    tmp=1
    #else:
    #    tmp=t
    tmp=t**alp
    return tmp


 
 
###########

