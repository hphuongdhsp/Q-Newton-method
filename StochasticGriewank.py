import math
import numpy as np
import numdifftools as nd
from numpy import linalg as LA
from random import random
import params as pr
import scipy
import scipy.special as scisp
from scipy.optimize import fmin_bfgs
import algopy
from algopy import UTPM, exp
from utils import *
import time, datetime
import cubic_reg2, cubic_reg

"""Here list all cost functions reported in experiments, in the stochastic setting.
Details for each function, its gradient and Hessian, as well as eigenvalues of the Hessian
"""

#### Importing parameters for functions

gamma = pr.gamma
a = pr.a
b = pr.b
c6 = pr.c6
c9 = pr.c9
coef = pr.coef
sign1 = pr.sign1
sign2 = pr.sign2
ep = pr.ep
atol=pr.atol
rtol=pr.rtol
#D=pr.D


######## Check for normal distribution #############



#s=np.random.normal(mu, sigma,10)









####### Stochastic versions of algorithms

#Newton's method:
def NewtonMethod():
    print("0.Newton's method:")

    z00=z00_old
    z00_1=z00_old
    #print(z00)
    time0=time.time()
    for m in range(NIterate):
        #print("Iteration =",m)
        xi=np.random.normal(mu, sigma,N1)
        tmp, w = fHessian(z00,xi)
        Der=fDer(z00,xi)
        if dimN == 1:
            HessTr = tmp
            HessInv = 1 / HessTr
            gn = Der * HessInv
        elif dimN > 1:
            HessTr = tmp.transpose()
            HessInv = LA.inv(HessTr)
            gn = np.matmul(Der, HessInv)
    
        z00 = z00 - gn
        if verbose:
            print(f(z00,xi))
            #print(z00)
            
        if stopCriterion == 0:
            if (L2Norm2(Der, dimN) < atol) :  # stop when meet the relative and absolute tolerances
                break
        elif stopCriterion == 2:
            if abs(f(z00,xi)-f(zmin,xi))<errtol:
                break
        if L2Norm2(z00-z00_1,dimN)<atol:
            break
        z00_1=z00
    time1=time.time()
    print("time=",time1-time0)
    print("m=", m)
    #print(z00)
    if printLastPoint==True:
        print("The last point =",z00)
    if verbose==False:
        print("function value =", f(z00,xi))
    print("norm of gradient=", math.sqrt(L2Norm2(fDer(z00,xi),dimN)))
    if printHessianEigenvalues==True:
        tmp, w=fHessian(z00,xi)
        #print(fDer(z00_old))
        print("Eigenvalues of Hessian=", w)
    #print(fDer(z00))
    return

def LocalNewtonMethod():
    print("0.Local.Newton's method:")

    z00=z00_old
    z00_1=z00_old
    #print(z00)
    time0=time.time()
    for m in range(NIterate):
        #print("Iteration =",m)
        xi=np.random.normal(mu, sigma,N1)
        tmp, w = fHessian(z00,xi)
        Der=fDer(z00,xi)
        if dimN == 1:
            HessTr = tmp
            HessInv = 1 / HessTr
            gn = Der * HessInv
        elif dimN > 1:
            HessTr = tmp.transpose()
            HessInv = LA.inv(HessTr)
            gn = np.matmul(Der, HessInv)
        kkapp=1
        while constraintChect(z00-kkapp*gn)==False:
            kkapp=kkapp/2
            
    
        z00 = z00 - kkapp*gn
        if verbose:
            print(f(z00,xi))
            #print(z00)
            
        if stopCriterion == 0:
            if (L2Norm2(Der, dimN) < atol) :  # stop when meet the relative and absolute tolerances
                break
        elif stopCriterion == 2:
            if abs(f(z00,xi)-f(zmin,xi))<errtol:
                break
        if L2Norm2(z00-z00_1,dimN)<aatol:
            break
        z00_1=z00
    time1=time.time()
    print("time=",time1-time0)
    print("m=", m)
    #print(z00)
    if printLastPoint==True:
        print("The last point =",z00)
    if verbose==False:
        print("function value =", f(z00,xi))
    print("norm of gradient=", math.sqrt(L2Norm2(fDer(z00,xi),dimN)))
    if printHessianEigenvalues==True:
        tmp, w=fHessian(z00,xi)
        #print(fDer(z00_old))
        print("Eigenvalues of Hessian=", w)
    #print(fDer(z00))
    return


#Random Newton's method:

def RandomNewtonMethod():

    print("1.Random Newton's method:")

    z00=z00_old
    z00_1=z00_old
    #print(z00)

    minLR = 0
    maxLR = 2

    time0=time.time()
    for m in range(NIterate):
        
        delta = minLR + random() * (maxLR - minLR)
        xi=np.random.normal(mu, sigma,N1)
        Der=fDer(z00,xi)
        tmp, w = fHessian(z00,xi)
        if dimN == 1:
            HessTr = tmp
            HessInv = 1 / HessTr
            gn = delta * Der * HessInv
        elif dimN > 1:
            HessTr = tmp.transpose()
            HessInv = LA.inv(HessTr)
            gn = delta * np.matmul(Der, HessInv)
    
        z00 = z00 - gn
        if verbose:
            print(f(z00,xi))
            #print(z00)
            
        if stopCriterion == 0:
            if (L2Norm2(Der, dimN) < atol) :  # stop when meet the relative and absolute tolerances
                break
        elif stopCriterion == 2:
            if abs(f(z00,xi)-f(zmin,xi))<errtol:
                break
        if L2Norm2(z00-z00_1,dimN)<aatol:
            break
        z00_1=z00
    time1=time.time()
    print("time=",time1-time0)
    print("m=", m)
    #print(z00)
    if printLastPoint==True:
        print("The last point =",z00)
    if verbose==False:
        print("function value =", f(z00,xi))
    print("norm of gradient=", math.sqrt(L2Norm2(fDer(z00,xi),dimN)))
    if printHessianEigenvalues==True:
        tmp, w=fHessian(z00,xi)
        #print(fDer(z00_old))
        print("Eigenvalues of Hessian=", w)
    #print(fDer(z00))
    return
    
    
def LocalRandomNewtonMethod():

    print("1.Local.Random Newton's method:")

    z00=z00_old
    z00_1=z00_old
    #print(z00)

    minLR = 0
    maxLR = 2

    time0=time.time()
    for m in range(NIterate):

        delta = minLR + random() * (maxLR - minLR)
        xi=np.random.normal(mu, sigma,N1)
        Der=fDer(z00,xi)
        tmp, w = fHessian(z00,xi)
        if dimN == 1:
            HessTr = tmp
            HessInv = 1 / HessTr
            gn = delta * Der * HessInv
        elif dimN > 1:
            HessTr = tmp.transpose()
            HessInv = LA.inv(HessTr)
            gn = delta * np.matmul(Der, HessInv)
    
        kkapp=1
        while constraintChect(z00-kkapp*gn)==False:
            kkapp=kkapp/2
            
    
        z00 = z00 - kkapp*gn
        if verbose:
            print(f(z00,xi))
            #print(z00)
            
        if stopCriterion == 0:
            if (L2Norm2(Der, dimN) < atol) :  # stop when meet the relative and absolute tolerances
                break
        elif stopCriterion == 2:
            if abs(f(z00,xi)-f(zmin,xi))<errtol:
                break
        if L2Norm2(z00-z00_1,dimN)<aatol:
            break
        z00_1=z00
    time1=time.time()
    print("time=",time1-time0)
    print("m=", m)
    #print(z00)
    if printLastPoint==True:
        print("The last point =",z00)
    if verbose==False:
        print("function value =", f(z00,xi))
    print("norm of gradient=", math.sqrt(L2Norm2(fDer(z00,xi),dimN)))
    if printHessianEigenvalues==True:
        tmp, w=fHessian(z00,xi)
        #print(fDer(z00_old))
        print("Eigenvalues of Hessian=", w)
    #print(fDer(z00))
    return
    
    
#NewQNewton

def NewQNewton():

    print("2. New Q Newton's method:")

    z00=z00_old
    z00_1=z00_old
    delta=1
    time0=time.time()
    #print(z00)
    for m in range(NIterate):
        #print("Iteration=",m)
        xi=np.random.normal(mu, sigma,N1)
        tmp, w=fHessian(z00,xi)
        Der=fDer(z00,xi)
        if dimN == 1:
            HessTr = tmp
            #print(HessTr)
            if abs(HessTr)==0:
                HessTr=HessTr+delta*cutoff(L2Norm2(Der,dimN))
                #print(HessTr)
        
            HessInv = 1 / HessTr
            #print(HessInv)
            gn = Der * HessInv
            gn=NegativeOrthogonalDecomposition(HessTr,gn,dimN)
        elif dimN > 1:
            HessTr = tmp.transpose()
            #print(HessTr)
            #print(f26Der(z00))
            if abs(LA.det(HessTr))==0:
                HessTr=HessTr+delta*cutoff(L2Norm2(Der,dimN))*np.identity(dimN,dtype=float)
                #print(HessTr)
            HessInv = LA.inv(HessTr)
            #print(HessInv)
            gn = np.matmul(Der, HessInv)
            gn=NegativeOrthogonalDecomposition(HessTr,gn,dimN)
            
    
        z00 = z00 - gn
        if verbose:
            print(f(z00,xi))
            #print(z00)
            
        if stopCriterion == 0:
            if (L2Norm2(Der, dimN) < atol) :  # stop when meet the relative and absolute tolerances
                break
        elif stopCriterion == 2:
            if abs(f(z00,xi)-f(zmin,xi))<errtol:
                break
        if L2Norm2(z00-z00_1,dimN)<aatol:
            break
        z00_1=z00
    time1=time.time()
    print("time=",time1-time0)
    print("m=", m)
    #print(z00)
    if printLastPoint==True:
        print("The last point =",z00)
    if verbose==False:
        print("function value =", f(z00,xi))
    print("norm of gradient=", math.sqrt(L2Norm2(fDer(z00,xi),dimN)))
    if printHessianEigenvalues==True:
        tmp, w=fHessian(z00,xi)
        #print(fDer(z00_old))
        print("Eigenvalues of Hessian=", w)
    #print(fDer(z00))
    return
    
    
    
def LocalNewQNewton():

    print("2.Local. New Q Newton's method:")

    z00=z00_old
    z00_1=z00_old
    delta=1
    time0=time.time()
    #print(z00)
    for m in range(NIterate):
        #print("Iteration=",m)
        xi=np.random.normal(mu, sigma,N1)
        tmp, w=fHessian(z00,xi)
        Der=fDer(z00,xi)
        if dimN == 1:
            HessTr = tmp
            #print(HessTr)
            if abs(HessTr)==0:
                HessTr=HessTr+delta*cutoff(L2Norm2(Der,dimN))
                #print(HessTr)
        
            HessInv = 1 / HessTr
            #print(HessInv)
            gn = Der * HessInv
            gn=NegativeOrthogonalDecomposition(HessTr,gn,dimN)
        elif dimN > 1:
            HessTr = tmp.transpose()
            #print(HessTr)
            #print(f26Der(z00))
            if abs(LA.det(HessTr))==0:
                HessTr=HessTr+delta*cutoff(L2Norm2(Der,dimN))*np.identity(dimN,dtype=float)
                #print(HessTr)
            HessInv = LA.inv(HessTr)
            #print(HessInv)
            gn = np.matmul(Der, HessInv)
            gn=NegativeOrthogonalDecomposition(HessTr,gn,dimN)
            
    
        kkapp=1
        while constraintChect(z00-kkapp*gn)==False:
            kkapp=kkapp/2
            
    
        z00 = z00 - kkapp*gn
        
        if verbose:
            print(f(z00))
            #print(z00)
            
        if stopCriterion == 0:
            if (L2Norm2(Der, dimN) < atol) :  # stop when meet the relative and absolute tolerances
                break
        elif stopCriterion == 2:
            if abs(f(z00,xi)-f(zmin,xi))<errtol:
                break
        if L2Norm2(z00-z00_1,dimN)<aatol:
            break
        z00_1=z00
    time1=time.time()
    print("time=",time1-time0)
    print("m=", m)
    #print(z00)
    if printLastPoint==True:
        print("The last point =",z00)
    if verbose==False:
        print("function value =", f(z00,xi))
    print("norm of gradient=", math.sqrt(L2Norm2(fDer(z00,xi),dimN)))
    if printHessianEigenvalues==True:
        tmp, w=fHessian(z00,xi)
        #print(fDer(z00_old))
        print("Eigenvalues of Hessian=", w)
    #print(fDer(z00))
    return
    
    
#Random New Q Newton's method

def RandomNewQNewton():

    print("3. Random New Q Newton's method:")
    z00=z00_old
    z00_1=z00_old
    #print(z00)
    time0=time.time()
    for m in range(NIterate):
        xi=np.random.normal(mu, sigma,N1)
        tmp, w = fHessian(z00,xi)
        Der=fDer(z00,xi)
        delta=random()
        if dimN == 1:
            HessTr = tmp
            if abs(HessTr)==0:
                HessTr=HessTr+delta*cutoff(L2Norm2(Der,dimN))
        
            HessInv = 1 / HessTr
            gn = Der * HessInv
            gn=NegativeOrthogonalDecomposition(HessTr,gn,dimN)
        elif dimN > 1:
            HessTr = tmp.transpose()
            if abs(LA.det(HessTr))==0:
                HessTr=HessTr+delta*cutoff(L2Norm2(Der,dimN))*np.identity(dimN,dtype=float)
            HessInv = LA.inv(HessTr)
            gn = np.matmul(Der, HessInv)
            gn=NegativeOrthogonalDecomposition(HessTr,gn,dimN)
    
        z00 = z00 - gn
        if verbose:
            print(f(z00))
            #print(z00)
            
        if stopCriterion == 0:
            if (L2Norm2(Der, dimN) < atol) :  # stop when meet the relative and absolute tolerances
                break
        elif stopCriterion == 2:
            if abs(f(z00,xi)-f(zmin,xi))<errtol:
                break
        if L2Norm2(z00-z00_1,dimN)<aatol:
            break
        z00_1=z00
    time1=time.time()
    print("time=",time1-time0)
    print("m=", m)
    #print(z00)
    if printLastPoint==True:
        print("The last point =",z00)
    if verbose==False:
        print("function value =", f(z00,xi))
    print("norm of gradient=", math.sqrt(L2Norm2(fDer(z00,xi),dimN)))
    if printHessianEigenvalues==True:
        tmp, w=fHessian(z00,xi)
        #print(fDer(z00_old))
        print("Eigenvalues of Hessian=", w)
    #print(fDer(z00))
    return

def LocalRandomNewQNewton():

    print("3.Local. Random New Q Newton's method:")
    z00=z00_old
    z00_1=z00_old
    #print(z00)
    time0=time.time()
    for m in range(NIterate):
        xi=np.random.normal(mu, sigma,N1)
        tmp, w = fHessian(z00,xi)
        Der=fDer(z00,xi)
        delta=random()
        if dimN == 1:
            HessTr = tmp
            if abs(HessTr)==0:
                HessTr=HessTr+delta*cutoff(L2Norm2(Der,dimN))
        
            HessInv = 1 / HessTr
            gn = Der * HessInv
            gn=NegativeOrthogonalDecomposition(HessTr,gn,dimN)
        elif dimN > 1:
            HessTr = tmp.transpose()
            if abs(LA.det(HessTr))==0:
                HessTr=HessTr+delta*cutoff(L2Norm2(Der,dimN))*np.identity(dimN,dtype=float)
            HessInv = LA.inv(HessTr)
            gn = np.matmul(Der, HessInv)
            gn=NegativeOrthogonalDecomposition(HessTr,gn,dimN)
    
        kkapp=1
        while constraintChect(z00-kkapp*gn)==False:
            kkapp=kkapp/2
            
    
        z00 = z00 - kkapp*gn
        if verbose:
            print(f(z00,xi))
            #print(z00)
            
        if stopCriterion == 0:
            if (L2Norm2(Der, dimN) < atol) :  # stop when meet the relative and absolute tolerances
                break
        elif stopCriterion == 2:
            if abs(f(z00,xi)-f(zmin,xi))<errtol:
                break
        if L2Norm2(z00-z00_1,dimN)<aatol:
            break
        z00_1=z00
    time1=time.time()
    print("time=",time1-time0)
    print("m=", m)
    #print(z00)
    if printLastPoint==True:
        print("The last point =",z00)
    if verbose==False:
        print("function value =", f(z00,xi))
    print("norm of gradient=", math.sqrt(L2Norm2(fDer(z00,xi),dimN)))
    if printHessianEigenvalues==True:
        tmp, w=fHessian(z00,xi)
        #print(fDer(z00_old))
        print("Eigenvalues of Hessian=", w)
    #print(fDer(z00))
    return


#BFGS


    
def LocalBFGS():

    print("4.Local. BFGS")
    z00=z00_old
    z00_1=z00_old
    #print(z00)
    time0=time.time()
    
    #xopt=fmin_bfgs(f, z00,fprime=fDer,maxiter=NIterate,gtol=pr.rtol, retall=True)
    
    for m in range(NIterate):
        xi=np.random.normal(mu, sigma,N1)
        g_x=fDer(z00,xi)
    
        xopt=fmin_bfgs(f, z00,fprime=fDer,maxiter=1,gtol=pr.rtol)
        kkapp=1
        while constraintChect(z00+kkapp*(xopt-z00))==False:
            kkapp=kkapp/2
        z00=z00+kkapp*(xopt-z00)
        if verbose:
            print(f(z00))
            #print(z00)
            
        if stopCriterion == 0:
            if (L2Norm2(g_x, dimN) < atol) :  # stop when meet the relative and absolute tolerances
                break
        elif stopCriterion == 2:
            if abs(f(z00,xi)-f(zmin,xi))<errtol:
                break
        
        if L2Norm2(z00-z00_1,dimN)<aatol:
            break
        z00_1=z00
    
    
    time1=time.time()
    print("time=",time1-time0)
    print("number of iterates=", m)
    
    if printLastPoint==True:
        print("The last point =",z00)
    print("function value=", f(z00,xi))
    print("norm of gradient=", math.sqrt(L2Norm2(fDer(z00,xi),dimN)))
    if printHessianEigenvalues==True:
        tmp, w=fHessian(z00,xi)
        #print(fDer(z00_old))
        print("Eigenvalues of Hessian=", w)
    
    
    
    
    
    return

#Adaptive cubic regularisation



#Local Adaptive cubic regularisation

def LocalAdaptiveCubicRegularisation():

    print("7.Local. Adaptive Cubic Regularisation")
    z00=z00_old
    z00_1=z00_old
    #print(z00)
    time0=time.time()
    
    for m in range(NIterate):
        xi=np.random.normal(mu, sigma,N1)
        g_x=fDer(z00,xi)
    
        #cr=cubic_reg.AdaptiveCubicReg(z00, f=f,gradient=None, hessian=None,hessian_update_method='exact', conv_tol=rtol, maxiter=1)
        #xopt, intermediate_points,  n_iter, flag=cr.adaptive_cubic_reg()
        cr2=cubic_reg2.AdaptiveCubicReg(z00, f=f,gradient=None, hessian=None,hessian_update_method='exact', conv_tol=rtol,maxiter=1)
        xopt, intermediate_points, intermediate_function_values,  n_iter, flag=cr2.adaptive_cubic_reg()
        kkapp=1
        while constraintChect(z00+kkapp*(xopt-z00))==False:
            kkapp=kkapp/2
        z00=z00+kkapp*(xopt-z00)
        if verbose:
            print(f(z00,xi))
            #print(z00)
            
        if stopCriterion == 0:
            if (L2Norm2(g_x, dimN) < atol) :  # stop when meet the relative and absolute tolerances
                break
        elif stopCriterion == 2:
            if abs(f(z00,xi)-f(zmin,xi))<errtol:
                break
        if L2Norm2(z00-z00_1,dimN)<aatol:
            break
        z00_1=z00
        
    
    
    time1=time.time()
    print("time=",time1-time0)
    print("number of iterates=", n_iter)
    #print("optimal point=", xopt)
    if printLastPoint==True:
        print("The last point =",z00)
    print("function value=", f(z00,xi))
    print("norm of gradient=", math.sqrt(L2Norm2(fDer(z00,xi),dimN)))
    if printHessianEigenvalues==True:
        tmp, w=fHessian(z00)
        #print(fDer(z00_old))
        print("Eigenvalues of Hessian=", w)
    
    return



#Backtraking Gradient Descent

def BacktrackingGD():

    print("5. Backtracking Gradient Descent")

    z00=z00_old
    z00_1=z00_old
    #print(z00)

    time0=time.time()
    delta0=pr.delta0
    alpha=pr.alpha
    beta=pr.beta

    for m in range(NIterate):
        xi=np.random.normal(mu, sigma,N1)
        delta = delta0
        x = z00
        f_x = f(x,xi)
        g_x = fDer(x,xi)
        gx_norm = L2Norm2(g_x, dimN)
        g_xx, w = fHessian(x,xi)
        gxx_norm = math.sqrt(L2Norm2(g_xx, dimN))
        x_check = x - delta * g_x
        check = f(x_check,xi) - f_x + alpha * delta * gx_norm
        count = 0

        if check > 0:
            while check > 0:  # and delta > 1e-6:
                count += 1
                delta = delta * beta  # rescale delta
                x_check = x - delta * g_x
                check = f(x_check,xi) - f_x + alpha * delta * gx_norm
    
        gn = delta * fDer(z00,xi)
        z00 = z00 - gn
        if verbose:
            print(f(z00,xi))
            #print(z00)
            
        if stopCriterion == 0:
            if (L2Norm2(g_x, dimN) < atol) :  # stop when meet the relative and absolute tolerances
                break
        elif stopCriterion == 2:
            if abs(f(z00,xi)-f(zmin,xi))<errtol:
                break
        if L2Norm2(z00-z00_1,dimN)<aatol:
            break
        z00_1=z00
    time1=time.time()
    print("time=",time1-time0)
    print("m=", m)
    #print(z00)
    if printLastPoint==True:
        print("The last point =",z00)
    if verbose==False:
        print("function value =", f(z00,xi))
    print("norm of gradient=", math.sqrt(L2Norm2(fDer(z00,xi),dimN)))
    if printHessianEigenvalues==True:
        tmp, w=fHessian(z00,xi)
        #print(fDer(z00_old))
        print("Eigenvalues of Hessian=", w)
    
    #print(fDer(z00))
    return

def LocalBacktrackingGD():

    print("5.Local. Backtracking Gradient Descent")

    z00=z00_old
    z00_1=z00_old
    #print(z00)

    time0=time.time()
    delta0=pr.delta0
    alpha=pr.alpha
    beta=pr.beta

    for m in range(NIterate):
        xi=np.random.normal(mu, sigma,N1)
        delta = delta0
        x = z00
        f_x = f(x,xi)
        g_x = fDer(x,xi)
        gx_norm = L2Norm2(g_x, dimN)
        g_xx, w = fHessian(x,xi)
        gxx_norm = math.sqrt(L2Norm2(g_xx, dimN))
        x_check = x - delta * g_x
        check = f(x_check) - f_x + alpha * delta * gx_norm
        count = 0

        if check > 0:
            while check > 0:  # and delta > 1e-6:
                count += 1
                delta = delta * beta  # rescale delta
                x_check = x - delta * g_x
                check = f(x_check) - f_x + alpha * delta * gx_norm
    
        gn = delta * fDer(z00)
        
        kkapp=1
        while constraintChect(z00-kkapp*gn)==False:
            kkapp=kkapp/2
            
    
        z00 = z00 - kkapp*gn
        
        if verbose:
            print(f(z00,xi))
            #print(z00)
            
        if stopCriterion == 0:
            if (L2Norm2(g_x, dimN) < atol) :  # stop when meet the relative and absolute tolerances
                break
        elif stopCriterion == 2:
            if abs(f(z00,xi)-f(zmin,xi))<errtol:
                break
        if L2Norm2(z00-z00_1,dimN)<aatol:
            break
        z00_1=z00
    time1=time.time()
    print("time=",time1-time0)
    print("m=", m)
    #print(z00)
    if printLastPoint==True:
        print("The last point =",z00)
    if verbose==False:
        print("function value =", f(z00,xi))
    print("norm of gradient=", math.sqrt(L2Norm2(fDer(z00,xi),dimN)))
    if printHessianEigenvalues==True:
        tmp, w=fHessian(z00,xi)
        #print(fDer(z00_old))
        print("Eigenvalues of Hessian=", w)
    
    #print(fDer(z00))
    return



#Two-way Backtracking

def TwoWayBacktrackingGD():
    print("8. Two-way Backtracking GD")
    
    z00=z00_old
    z00_1=z00_old
    
    
    time0=time.time()
    delta0=pr.delta0
    alpha=pr.alpha
    beta=pr.beta
    
    
    
    lr_pure = [delta0]
    productGraph=[]
    

    print("Number of maximum iterates:", NIterate, "; But note that the number of actual iterates may be much smaller")

    for m in range(NIterate):
        xi=np.random.normal(mu, sigma,N1)
        delta = lr_pure[-1]
        #    delta_tmp = delta
        x = z00
        f_x = f(x,xi)
        g_x = fDer(x,xi)
        gx_norm = L2Norm2(g_x, dimN)
        g_xx, w=fHessian(x,xi)
        gxx_norm=math.sqrt(L2Norm2(g_xx, dimN))
        #x_prev = x
        x_check = x - delta * g_x
        check = f(x_check) - f_x + alpha * delta * gx_norm
        count = 0
        
        #    if avoid == 0:
        if check > 0:
            while check > 0:  # and delta > 1e-6:
                count += 1;
                delta = delta * beta  # rescale delta
                #x_prev = x_check
                x_check = x - delta * g_x
                check = f(x_check,xi) - f_x + alpha * delta * gx_norm
        elif check < 0:
            while check < 0 and delta <= delta0:
                count += 1
                delta = delta / beta  # rescale delta
                x_check = x - delta * g_x
                check = f(x - delta * g_x,xi) - f_x + alpha * delta * gx_norm
            delta = delta * beta

        lr_pure.append(delta)
        #productGraph.append(delta*gxx_norm)
        
        gn = delta * fDer(z00,xi)
        z00 = z00 - gn
        if verbose:
            print(f(z00,xi))
            #print(z00)
            
        if stopCriterion == 0:
            if (L2Norm2(g_x, dimN) < atol) :  # stop when meet the relative and absolute tolerances
                break
        elif stopCriterion == 2:
            if abs(f(z00,xi)-f(zmin,xi))<errtol:
                break
        if L2Norm2(z00-z00_1,dimN)<aatol:
            break
        z00_1=z00
    time1=time.time()
    print("time=",time1-time0)
    print("m=", m)
    #print(z00)
    if printLastPoint==True:
        print("The last point =",z00)
    if verbose==False:
        print("function value =", f(z00,xi))
    print("norm of gradient=", math.sqrt(L2Norm2(fDer(z00,xi),dimN)))
    if printHessianEigenvalues==True:
        tmp, w=fHessian(z00,xi)
        #print(fDer(z00_old))
        print("Eigenvalues of Hessian=", w)
    
   
    #print(fDer(z00))
    return

def LocalTwoWayBacktrackingGD():
    print("8.Local. Two-way Backtracking GD")
    
    z00=z00_old
    z00_1=z00_old
    
    
    time0=time.time()
    delta0=pr.delta0
    alpha=pr.alpha
    beta=pr.beta
    
    
    
    lr_pure = [delta0]
    productGraph=[]
    

    print("Number of maximum iterates:", NIterate, "; But note that the number of actual iterates may be much smaller")

    for m in range(NIterate):
        xi=np.random.normal(mu, sigma,N1)
        delta = lr_pure[-1]
        #    delta_tmp = delta
        x = z00
        f_x = f(x,xi)
        g_x = fDer(x,xi)
        gx_norm = L2Norm2(g_x, dimN)
        g_xx, w=fHessian(x,xi)
        gxx_norm=math.sqrt(L2Norm2(g_xx, dimN))
        #x_prev = x
        x_check = x - delta * g_x
        check = f(x_check,xi) - f_x + alpha * delta * gx_norm
        count = 0
        
        #    if avoid == 0:
        if check > 0:
            while check > 0:  # and delta > 1e-6:
                count += 1;
                delta = delta * beta  # rescale delta
                #x_prev = x_check
                x_check = x - delta * g_x
                check = f(x_check,xi) - f_x + alpha * delta * gx_norm
        elif check < 0:
            while check < 0 and delta <= delta0:
                count += 1
                delta = delta / beta  # rescale delta
                x_check = x - delta * g_x
                check = f(x - delta * g_x,xi) - f_x + alpha * delta * gx_norm
            delta = delta * beta

        lr_pure.append(delta)
        #productGraph.append(delta*gxx_norm)
        
        gn = delta * fDer(z00,xi)
        
        kkapp=1
        while constraintChect(z00-kkapp*gn)==False:
            kkapp=kkapp/2
            
    
        z00 = z00 - kkapp*gn
        
        if verbose:
            print(f(z00,xi))
            #print(z00)
            
        if stopCriterion == 0:
            if (L2Norm2(g_x, dimN) < atol) :  # stop when meet the relative and absolute tolerances
                break
        elif stopCriterion == 2:
            if abs(f(z00,xi)-f(zmin,xi))<errtol:
                break
        if L2Norm2(z00-z00_1,dimN)<aatol:
            break
        z00_1=z00
    time1=time.time()
    print("time=",time1-time0)
    print("m=", m)
    #print(z00)
    if printLastPoint==True:
        print("The last point =",z00)
    if verbose==False:
        print("function value =", f(z00,xi))
    print("norm of gradient=", math.sqrt(L2Norm2(fDer(z00,xi),dimN)))
    if printHessianEigenvalues==True:
        tmp, w=fHessian(z00,xi)
        #print(fDer(z00_old))
        print("Eigenvalues of Hessian=", w)
    
   
    #print(fDer(z00))
    return



#Unbounded Two-way Backtracking GD

def UnboundedTwoWayBacktrackingGD():
    print("9. Unbounded Two-way Backtracking GD")
    
    z00=z00_old
    z00_1=z00_old
    
    
    time0=time.time()
    delta0=pr.delta0
    alpha=pr.alpha
    beta=pr.beta
    
    
    
    lr_pure = [delta0]
    productGraph=[]
    

    print("Number of maximum iterates:", NIterate, "; But note that the number of actual iterates may be much smaller")

    for m in range(NIterate):
        xi=np.random.normal(mu, sigma,N1)
        delta = lr_pure[-1]
        #    delta_tmp = delta
        x = z00
        #print("the value of xi", xi)
        #print("the value at [1,1]",f1([1,1],xi))
        f_x = f(x,xi)
        g_x = fDer(x,xi)
        gx_norm = L2Norm2(g_x, dimN)
        gx_normSquareRoot=math.sqrt(gx_norm)
        g_xx, w=fHessian(x,xi)
        gxx_norm=math.sqrt(L2Norm2(g_xx, dimN))
        #x_prev = x
        x_check = x - delta * g_x
        check = f(x_check,xi) - f_x + alpha * delta * gx_norm
        count = 0
        
        #    if avoid == 0:
        if check > 0:
            while check > 0:  # and delta > 1e-6:
                count += 1;
                delta = delta * beta  # rescale delta
                #x_prev = x_check
                x_check = x - delta * g_x
                check = f(x_check,xi) - f_x + alpha * delta * gx_norm
        elif check < 0:
            while check < 0 and delta <= UnboundedLR(gx_normSquareRoot,1e+100, delta0):
                count += 1
                delta = delta / beta  # rescale delta
                x_check = x - delta * g_x
                check = f(x - delta * g_x,xi) - f_x + alpha * delta * gx_norm
            delta = delta * beta

        lr_pure.append(delta)
        #productGraph.append(delta*gxx_norm)
        
        gn = delta * fDer(z00,xi)
        z00 = z00 - gn
        if verbose:
            print(f(z00,xi))
            #print(z00)
            
        if stopCriterion == 0:
            if (L2Norm2(g_x, dimN) < atol) :  # stop when meet the relative and absolute tolerances
                break
        elif stopCriterion == 2:
            if abs(f(z00,xi)-f(zmin,xi))<errtol:
                break
        if L2Norm2(z00-z00_1,dimN)<aatol:
            break
        z00_1=z00
    time1=time.time()
    print("time=",time1-time0)
    print("m=", m)
    #print(z00)
    if printLastPoint==True:
        print("The last point =",z00)
    if verbose==False:
        print("function value =", f(z00,xi))
    
    #print(fDer(z00))
    flag=0
    for i in range(len(lr_pure)):
        if delta0<lr_pure[i]:
            flag+=1
    print("Number of learning rates bigger than \delta _0=", flag)
    print("norm of gradient=", math.sqrt(L2Norm2(fDer(z00,xi),dimN)))
    if printHessianEigenvalues==True:
        tmp, w=fHessian(z00,xi)
        #print(fDer(z00_old))
        print("Eigenvalues of Hessian=", w)
    return


def LocalUnboundedTwoWayBacktrackingGD():
    print("9.Local. Unbounded Two-way Backtracking GD")
    
    z00=z00_old
    z00_1=z00_old
    
    
    time0=time.time()
    delta0=pr.delta0
    alpha=pr.alpha
    beta=pr.beta
    
    
    
    lr_pure = [delta0]
    productGraph=[]
    

    print("Number of maximum iterates:", NIterate, "; But note that the number of actual iterates may be much smaller")

    for m in range(NIterate):
        xi=np.random.normal(mu, sigma,N1)
        delta = lr_pure[-1]
        #    delta_tmp = delta
        x = z00
        f_x = f(x,xi)
        g_x = fDer(x,xi)
        gx_norm = L2Norm2(g_x, dimN)
        gx_normSquareRoot=math.sqrt(gx_norm)
        g_xx, w=fHessian(x,xi)
        gxx_norm=math.sqrt(L2Norm2(g_xx, dimN))
        #x_prev = x
        x_check = x - delta * g_x
        check = f(x_check,xi) - f_x + alpha * delta * gx_norm
        count = 0
        
        #    if avoid == 0:
        if check > 0:
            while check > 0:  # and delta > 1e-6:
                count += 1;
                delta = delta * beta  # rescale delta
                #x_prev = x_check
                x_check = x - delta * g_x
                check = f(x_check,xi) - f_x + alpha * delta * gx_norm
        elif check < 0:
            while check < 0 and delta <= UnboundedLR(gx_normSquareRoot,1e+100, delta0):
                count += 1
                delta = delta / beta  # rescale delta
                x_check = x - delta * g_x
                check = f(x - delta * g_x,xi) - f_x + alpha * delta * gx_norm
            delta = delta * beta

        lr_pure.append(delta)
        #productGraph.append(delta*gxx_norm)
        
        gn = delta * fDer(z00,xi)
        
        kkapp=1
        while constraintChect(z00-kkapp*gn)==False:
            kkapp=kkapp/2
            
    
        z00 = z00 - kkapp*gn
        
        if verbose:
            print(f(z00,xi))
            #print(z00)
            
        if stopCriterion == 0:
            if (L2Norm2(g_x, dimN) < atol) :  # stop when meet the relative and absolute tolerances
                break
        elif stopCriterion == 2:
            if abs(f(z00,xi)-f(zmin,xi))<errtol:
                break
        if L2Norm2(z00-z00_1,dimN)<aatol:
            break
        z00_1=z00
    time1=time.time()
    print("time=",time1-time0)
    print("m=", m)
    #print(z00)
    if printLastPoint==True:
        print("The last point =",z00)
    if verbose==False:
        print("function value =", f(z00,xi))
    
    #print(fDer(z00))
    flag=0
    for i in range(len(lr_pure)):
        if delta0<lr_pure[i]:
            flag+=1
    print("Number of learning rates bigger than \delta _0=", flag)
    print("norm of gradient=", math.sqrt(L2Norm2(fDer(z00),dimN)))
    if printHessianEigenvalues==True:
        tmp, w=fHessian(z00,xi)
        #print(fDer(z00_old))
        print("Eigenvalues of Hessian=", w)
    return


def InertialNewtonM():
    print("6. Inertial Newton's method")

    z00=z00_old
    z00_1=z00_old

    #print(z00)

    alpha1 = 0.5

    beta1 = 0.1

    lambda1 = (1/beta1) -alpha1



    psin=np.array([-1+random()*2 for _ in range(dimN)])



    time0=time.time()


    for m in range(NIterate):
        xi=np.random.normal(mu, sigma,N1)

        gn = fDer(z00,xi)

        gamman=(m+1)**(-0.5)

    

        z00 = z00 + gamman * (lambda1*z00-(1/beta1)*psin-beta1*gn)

        psin=psin+gamman*(lambda1*z00-(1/beta1)*psin)
        if verbose:
            print(f(z00,xi))
            #print(z00)
            
        if stopCriterion == 0:
            if (L2Norm2(gn, dimN) < atol) :  # stop when meet the relative and absolute tolerances
                break
        elif stopCriterion == 2:
            if abs(f(z00,xi)-f(zmin,xi))<errtol:
                break
        if L2Norm2(z00-z00_1,dimN)<aatol:
            break
        z00_1=z00
    time1=time.time()
    print("time=",time1-time0)
    print("m=", m)
    #print(z00)
    if printLastPoint==True:
        print("The last point =",z00)
    if verbose==False:
        print("function value =", f(z00,xi))
    print("norm of gradient=", math.sqrt(L2Norm2(fDer(z00,xi),dimN)))
    if printHessianEigenvalues==True:
        tmp, w=fHessian(z00,xi)
        #print(fDer(z00_old))
        print("Eigenvalues of Hessian=", w)
    return



#Local Inertial Newton's method

def LocalInertialNewtonM():
    print("6.Local. Inertial Newton's method")

    z00=z00_old
    z00_1=z00_old

    #print(z00)

    alpha1 = 0.5

    beta1 = 0.1

    lambda1 = (1/beta1) -alpha1



    psin=np.array([-1+random()*2 for _ in range(dimN)])



    time0=time.time()


    for m in range(NIterate):
        xi=np.random.normal(mu, sigma,N1)

        gn = fDer(z00,xi)

        gamman=(m+1)**(-0.5)
        
        ggn=gamman * (lambda1*z00-(1/beta1)*psin-beta1*gn)
        
        kkapp=1
        while constraintChect(z00+kkapp*ggn)==False:
            kkapp=kkapp/2
            
    
        z00 = z00 + kkapp*ggn

        

        psin=psin+gamman*(lambda1*z00-(1/beta1)*psin)
        if verbose:
            print(f(z00,xi))
            #print(z00)
            
        if stopCriterion == 0:
            if (L2Norm2(gn, dimN) < atol) :  # stop when meet the relative and absolute tolerances
                break
        elif stopCriterion == 2:
            if abs(f(z00,xi)-f(zmin,xi))<errtol:
                break
        if L2Norm2(z00-z00_1,dimN)<aatol:
            break
        z00_1=z00
    time1=time.time()
    print("time=",time1-time0)
    print("m=", m)
    #print(z00)
    if printLastPoint==True:
        print("The last point =",z00)
    if verbose==False:
        print("function value =", f(z00,xi))
    print("norm of gradient=", math.sqrt(L2Norm2(fDer(z00,xi),dimN)))
    if printHessianEigenvalues==True:
        tmp, w=fHessian(z00,xi)
        #print(fDer(z00_old))
        print("Eigenvalues of Hessian=", w)
    return
    



########### f3: Stochastic version of Griewank problem #################
########


def f3S(z,t):
    normz=np.sum(z**2)
    #print("normz=",normz)
    countN=np.array([t*z[i]/math.sqrt(i+1) for i in range(dimN)])
    productCos=1
    for i in range(dimN):
        productCos=productCos*math.cos(countN[i])
    tmp=1+(t*t)*(1/4000)*normz-productCos
    tmp=tmp
    #print("check what xi", xi)
    return tmp


def f3DerS(z,t):
    f3GS=nd.Gradient(f3S)
    tmp=f3GS(z,t)
    tmp1=tmp[0:dimN]
    return tmp1

def f3HessianS(z,t):
    f3HS=nd.Hessian(f3S)
    tmp=f3HS(z,t)
    tmp1=tmp[0:dimN,0:dimN]
    #w,v=LA.eig(tmp1)
    return tmp1

def f3(z,xi):
    tempArray=np.array([0 for _ in range(N1)])
    for i in range(N1):
        tempArray[i]=f3S(z,xi[i])
    tmp=np.sum(tempArray)
    tmp=tmp/N1
    #print("check what xi", xi)
    return tmp


def f3Der(z,xi):
    
    tempArray=np.array([0 for _ in range(dimN)])
    for i in range(N1):
        tempArray=tempArray+f3DerS(z,xi[i])
    tmp1=tempArray/N1
    
    return tmp1

def f3Hessian(z,xi):
    
    tempArray=np.array([[0 for _ in range(dimN)] for _ in range(dimN)])
    
    for i in range(N1):
        tmp1=f3HessianS(z,xi[i])
        tempArray=tempArray+tmp1
    
    tmp2=tempArray/N1
    w,v=LA.eig(tmp2)
    #w=0
    return tmp2, w


########### f3a: Stochastic version of Griewank problem, dim=1 #################
########




def f3a(z,xi):
    z1=z
    
    tmp1=0
    for i in range(N1):
        tmp1=tmp1+1+(1/4000)*((xi[i]*z1)**2)-math.cos(xi[i]*z1/math.sqrt(1))
    
    tmp=tmp1/N1
    #print("check what xi", xi)
    return tmp


def f3aDer(z,xi):
    
    f3aG=nd.Gradient(f3a)
    tmp=f3aG(z,xi)
    tmp1=tmp
    return tmp

def f3aHessian(z,xi):
    f3aH=nd.Hessian(f3a)
    tmp=f3aH(z,xi)
    tmp1=tmp
    
    
    w,v=LA.eig(tmp)
    #w=0
    return tmp1, w


########### f3b: Stochastic version of Griewank problem, dim=2 #################
########




def f3b(z,xi):
    z1,z2=z
    
    tmp1=0
    for i in range(N1):
        tmp1=tmp1+1+(1/4000)*((xi[i]*z1)**2)+(1/4000)*((xi[i]*z2)**2)-math.cos(xi[i]*z1/math.sqrt(1))*math.cos(xi[i]*z2/math.sqrt(2))
    
    tmp=tmp1/N1
    #print("check what xi", xi)
    return tmp


def f3bDer(z,xi):
    
    f3bG=nd.Gradient(f3b)
    tmp=f3bG(z,xi)
    tmp1=tmp
    return tmp

def f3bHessian(z,xi):
    f3bH=nd.Hessian(f3b)
    tmp=f3bH(z,xi)
    tmp1=tmp
    
    
    w,v=LA.eig(tmp)
    #w=0
    return tmp1, w

########### f3c: Stochastic version of Griewank problem, dim=10 #################
########




def f3c(z,xi):
    z1,z2,z3,z4,z5,z6,z7,z8,z9,z10=z
    
    tmp1=0
    for i in range(N1):
        tmp1=tmp1+1+(1/4000)*((xi[i]*z1)**2)+(1/4000)*((xi[i]*z2)**2)+(1/4000)*((xi[i]*z3)**2)+(1/4000)*((xi[i]*z4)**2)+(1/4000)*((xi[i]*z5)**2)+(1/4000)*((xi[i]*z6)**2)+(1/4000)*((xi[i]*z7)**2)+(1/4000)*((xi[i]*z8)**2)+(1/4000)*((xi[i]*z9)**2)+(1/4000)*((xi[i]*z10)**2)-math.cos(xi[i]*z1/math.sqrt(1))*math.cos(xi[i]*z2/math.sqrt(2))*math.cos(xi[i]*z3/math.sqrt(3))*math.cos(xi[i]*z4/math.sqrt(4))*math.cos(xi[i]*z5/math.sqrt(5))*math.cos(xi[i]*z6/math.sqrt(6))*math.cos(xi[i]*z7/math.sqrt(7))*math.cos(xi[i]*z8/math.sqrt(8))*math.cos(xi[i]*z9/math.sqrt(9))*math.cos(xi[i]*z10/math.sqrt(10))
    
    tmp=tmp1/N1
    #print("check what xi", xi)
    return tmp


def f3cDer(z,xi):
    
    f3cG=nd.Gradient(f3c)
    tmp=f3cG(z,xi)
    tmp1=tmp
    return tmp

def f3cHessian(z,xi):
    f3cH=nd.Hessian(f3c)
    tmp=f3cH(z,xi)
    tmp1=tmp
    
    
    w,v=LA.eig(tmp)
    #w=0
    return tmp1, w

########### f3d: Stochastic version of Griewank problem, dim=5 #################
########




def f3d(z,xi):
    z1,z2,z3,z4,z5=z
    
    tmp1=0
    for i in range(N1):
        tmp1=tmp1+1+(1/4000)*((xi[i]*z1)**2)+(1/4000)*((xi[i]*z2)**2)+(1/4000)*((xi[i]*z3)**2)+(1/4000)*((xi[i]*z4)**2)+(1/4000)*((xi[i]*z5)**2)-math.cos(xi[i]*z1/math.sqrt(1))*math.cos(xi[i]*z2/math.sqrt(2))*math.cos(xi[i]*z3/math.sqrt(3))*math.cos(xi[i]*z4/math.sqrt(4))*math.cos(xi[i]*z5/math.sqrt(5))
    
    tmp=tmp1/N1
    #print("check what xi", xi)
    return tmp


def f3dDer(z,xi):
    
    f3dG=nd.Gradient(f3d)
    tmp=f3dG(z,xi)
    tmp1=tmp
    return tmp

def f3dHessian(z,xi):
    f3dH=nd.Hessian(f3d)
    tmp=f3dH(z,xi)
    tmp1=tmp
    
    
    w,v=LA.eig(tmp)
    #w=0
    return tmp1, w


##############


mu, sigma =1, math.sqrt(1)

N1=10


xi=np.random.normal(mu, sigma,N1)
print(type(xi))

D=10
mm=1

dimN=D


bound=5

ggam=1






#verbose=True
#stopCriterion=0

printLastPoint=True

printHessianEigenvalues=True

verbose=False
#stopCriterion=0
stopCriterion=0
#stopCriterion=2

NIterate=1000
#NIterate=10000

atol=1e-12
print("atol=", atol)
aatol=1e-12

print("the function is", "f3c")
f=f3c
fDer=f3cDer
fHessian=f3cHessian


print("xi=", xi)


z00_old=np.array([10 for _ in range(dimN)])
#z00_old=np.array([20*np.random.rand()-10 for _ in range(dimN)])
print("Number of iterates=", NIterate)

#print("type=", type(f3S(z00_old,1)))

print("initial point=", z00_old)

print("function value at initial point=", f(z00_old,xi))
print("derivative at the initial point=", fDer(z00_old,xi))
#tmp=fHessian(z00_old)
tmp, w=fHessian(z00_old,xi)
print("Hessian at the initial point", tmp)
#print("Eigenvalues of the Hessian at the initial point=", w)

#print("f3S at xi[0]", f3S(z00_old,xi[0]))


NewtonMethod()
RandomNewQNewton()

RandomNewtonMethod()
NewQNewton()

InertialNewtonM()
UnboundedTwoWayBacktrackingGD()
#BacktrackingGD()

#TwoWayBacktrackingGD()



#LocalNewtonMethod()
#LocalRandomNewQNewton()
#LocalRandomNewtonMethod()
#LocalNewQNewton()

#LocalInertialNewtonM()
#LocalBacktrackingGD()

#LocalTwoWayBacktrackingGD()
#LocalUnboundedTwoWayBacktrackingGD()
#LocalBFGS()
#LocalAdaptiveCubicRegularisation()
#AdaptiveCubicRegularisation()




def mainmain():
    #N=10
    #N1=100
    z=np.array([1,1])
    xi=np.random.normal(mu, sigma,N1)
    print(xi)
    print(type(xi))
    print(f1(z,xi))
    print(f1Der(z,xi))
    #print(f1Hessian(z))
    #z=np.array([1,1])
    xi=np.random.normal(mu, sigma,N1)
    print(xi)
    print(type(xi))
    print(f1(z,xi))
    print(f1Der(z,xi))
    #print(f1Hessian(z))
    xi=np.random.normal(mu, sigma,N1)
    print(xi)
    print(type(xi))
    print(f1(z,xi))
    print(f1Der(z,xi))
    #print(f1Hessian(z))
    return
    
#mainmain()
    
