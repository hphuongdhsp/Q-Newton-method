#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 26 22:33:06 2021

@author: tdto
"""

import math
import numpy as np
import numdifftools as nd
import random
import params as pr

from numpy import linalg as LA
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


####### Optimisation algorithms

#Newton's method:
def NewtonMethod(f,fDer,fHessian,z00_old ,NIterate,dimN,verbose, stopCriterion):
    print("0.Newton's method:")

    z00=z00_old
    z00_1=z00_old
    #print(z00)
    time0=time.time()
    for m in range(NIterate):
        #print("Iteration =",m)
        tmp, w = fHessian(z00)
        Der=fDer(z00)
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
            print(f(z00))
            #print(z00)
            
        if stopCriterion == 0:
            if (L2Norm2(Der, dimN) < atol) :  # stop when meet the relative and absolute tolerances
                break
        elif stopCriterion == 2:
            if abs(f(z00)-f(zmin))<errtol:
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
        print("function value =", f(z00))
    print("norm of gradient=", math.sqrt(L2Norm2(fDer(z00),dimN)))
    if printHessianEigenvalues==True:
        tmp, w=fHessian(z00)
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
        tmp, w = fHessian(z00)
        Der=fDer(z00)
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
            print(f(z00))
            #print(z00)
            
        if stopCriterion == 0:
            if (L2Norm2(Der, dimN) < atol) :  # stop when meet the relative and absolute tolerances
                break
        elif stopCriterion == 2:
            if abs(f(z00)-f(zmin))<errtol:
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
        print("function value =", f(z00))
    print("norm of gradient=", math.sqrt(L2Norm2(fDer(z00),dimN)))
    if printHessianEigenvalues==True:
        tmp, w=fHessian(z00)
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
        Der=fDer(z00)
        tmp, w = fHessian(z00)
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
            print(f(z00))
            #print(z00)
            
        if stopCriterion == 0:
            if (L2Norm2(Der, dimN) < atol) :  # stop when meet the relative and absolute tolerances
                break
        elif stopCriterion == 2:
            if abs(f(z00)-f(zmin))<errtol:
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
        print("function value =", f(z00))
    print("norm of gradient=", math.sqrt(L2Norm2(fDer(z00),dimN)))
    if printHessianEigenvalues==True:
        tmp, w=fHessian(z00)
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
        Der=fDer(z00)
        tmp, w = fHessian(z00)
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
            print(f(z00))
            #print(z00)
            
        if stopCriterion == 0:
            if (L2Norm2(Der, dimN) < atol) :  # stop when meet the relative and absolute tolerances
                break
        elif stopCriterion == 2:
            if abs(f(z00)-f(zmin))<errtol:
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
        print("function value =", f(z00))
    print("norm of gradient=", math.sqrt(L2Norm2(fDer(z00),dimN)))
    if printHessianEigenvalues==True:
        tmp, w=fHessian(z00)
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
        tmp, w=fHessian(z00)
        Der=fDer(z00)
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
            print(f(z00))
            #print(z00)
            
        if stopCriterion == 0:
            if (L2Norm2(Der, dimN) < atol) :  # stop when meet the relative and absolute tolerances
                break
        elif stopCriterion == 2:
            if abs(f(z00)-f(zmin))<errtol:
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
        print("function value =", f(z00))
    print("norm of gradient=", math.sqrt(L2Norm2(fDer(z00),dimN)))
    if printHessianEigenvalues==True:
        tmp, w=fHessian(z00)
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
        tmp, w=fHessian(z00)
        Der=fDer(z00)
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
            if abs(f(z00)-f(zmin))<errtol:
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
        print("function value =", f(z00))
    print("norm of gradient=", math.sqrt(L2Norm2(fDer(z00),dimN)))
    if printHessianEigenvalues==True:
        tmp, w=fHessian(z00)
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
        tmp, w = fHessian(z00)
        Der=fDer(z00)
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
            if abs(f(z00)-f(zmin))<errtol:
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
        print("function value =", f(z00))
    print("norm of gradient=", math.sqrt(L2Norm2(fDer(z00),dimN)))
    if printHessianEigenvalues==True:
        tmp, w=fHessian(z00)
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
        tmp, w = fHessian(z00)
        Der=fDer(z00)
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
            print(f(z00))
            #print(z00)
            
        if stopCriterion == 0:
            if (L2Norm2(Der, dimN) < atol) :  # stop when meet the relative and absolute tolerances
                break
        elif stopCriterion == 2:
            if abs(f(z00)-f(zmin))<errtol:
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
        print("function value =", f(z00))
    print("norm of gradient=", math.sqrt(L2Norm2(fDer(z00),dimN)))
    if printHessianEigenvalues==True:
        tmp, w=fHessian(z00)
        #print(fDer(z00_old))
        print("Eigenvalues of Hessian=", w)
    #print(fDer(z00))
    return


#BFGS

def BFGS():

    print("4. BFGS")
    z00=z00_old
    #print(z00)
    time0=time.time()
    xopt=fmin_bfgs(f, z00,fprime=fDer,maxiter=NIterate,gtol=pr.rtol)
    #xopt=fmin_bfgs(f, z00,fprime=fDer,maxiter=NIterate,gtol=pr.rtol, retall=True)
    time1=time.time()
    print("time=",time1-time0)
    #print(xopt)
    if printLastPoint==True:
        print("The last point =",xopt)
    print("function value=",f(xopt))
    print("norm of gradient=", math.sqrt(L2Norm2(fDer(xopt),dimN)))
    if printHessianEigenvalues==True:
        tmp, w=fHessian(z00)
        #print(fDer(z00_old))
        print("Eigenvalues of Hessian=", w)
    return
    
def LocalBFGS():

    print("4.Local. BFGS")
    z00=z00_old
    z00_1=z00_old
    #print(z00)
    time0=time.time()
    
    #xopt=fmin_bfgs(f, z00,fprime=fDer,maxiter=NIterate,gtol=pr.rtol, retall=True)
    
    for m in range(NIterate):
        g_x=fDer(z00)
    
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
            if abs(f(z00)-f(zmin))<errtol:
                break
        
        if L2Norm2(z00-z00_1,dimN)<aatol:
            break
        z00_1=z00
    
    
    time1=time.time()
    print("time=",time1-time0)
    print("number of iterates=", m)
    
    if printLastPoint==True:
        print("The last point =",z00)
    print("function value=", f(z00))
    print("norm of gradient=", math.sqrt(L2Norm2(fDer(z00),dimN)))
    if printHessianEigenvalues==True:
        tmp, w=fHessian(z00)
        #print(fDer(z00_old))
        print("Eigenvalues of Hessian=", w)
    
    
    
    
    
    return

#Adaptive cubic regularisation

def AdaptiveCubicRegularisation():

    print("7. Adaptive Cubic Regularisation")
    z00=z00_old
    #print(z00)
    time0=time.time()
    
    cr=cubic_reg.AdaptiveCubicReg(z00, f=f,gradient=None, hessian=None,hessian_update_method='exact', conv_tol=rtol)
    xopt, intermediate_points,  n_iter, flag=cr.adaptive_cubic_reg()
    
    #cr2=cubic_reg2.AdaptiveCubicReg(z00, f=f,gradient=None, hessian=None,hessian_update_method='exact', conv_tol=rtol)
    #xopt, intermediate_points, intermediate_function_values,  n_iter, flag=cr2.adaptive_cubic_reg()
    
    time1=time.time()
    print("time=",time1-time0)
    print("number of iterates=", n_iter)
    #print("optimal point=", xopt)
    if printLastPoint==True:
        print("The last point =",xopt)
    print("function value=", f(xopt))
    print("norm of gradient=", math.sqrt(L2Norm2(fDer(xopt),dimN)))
    #print("function values of intermediate points=", intermediate_function_values)
    if printHessianEigenvalues==True:
        tmp, w=fHessian(z00)
        #print(fDer(z00_old))
        print("Eigenvalues of Hessian=", w)

    return


#Local Adaptive cubic regularisation

def LocalAdaptiveCubicRegularisation():

    print("7.Local. Adaptive Cubic Regularisation")
    z00=z00_old
    z00_1=z00_old
    #print(z00)
    time0=time.time()
    
    for m in range(NIterate):
        g_x=fDer(z00)
    
        #cr=cubic_reg.AdaptiveCubicReg(z00, f=f,gradient=None, hessian=None,hessian_update_method='exact', conv_tol=rtol, maxiter=1)
        #xopt, intermediate_points,  n_iter, flag=cr.adaptive_cubic_reg()
        cr2=cubic_reg2.AdaptiveCubicReg(z00, f=f,gradient=None, hessian=None,hessian_update_method='exact', conv_tol=rtol,maxiter=1)
        xopt, intermediate_points, intermediate_function_values,  n_iter, flag=cr2.adaptive_cubic_reg()
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
            if abs(f(z00)-f(zmin))<errtol:
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
    print("function value=", f(z00))
    print("norm of gradient=", math.sqrt(L2Norm2(fDer(z00),dimN)))
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
        delta = delta0
        x = z00
        f_x = f(x)
        g_x = fDer(x)
        gx_norm = L2Norm2(g_x, dimN)
        g_xx, w = fHessian(x)
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
        z00 = z00 - gn
        if verbose:
            print(f(z00))
            #print(z00)
            
        if stopCriterion == 0:
            if (L2Norm2(g_x, dimN) < atol) :  # stop when meet the relative and absolute tolerances
                break
        elif stopCriterion == 2:
            if abs(f(z00)-f(zmin))<errtol:
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
        print("function value =", f(z00))
    print("norm of gradient=", math.sqrt(L2Norm2(fDer(z00),dimN)))
    if printHessianEigenvalues==True:
        tmp, w=fHessian(z00)
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
        delta = delta0
        x = z00
        f_x = f(x)
        g_x = fDer(x)
        gx_norm = L2Norm2(g_x, dimN)
        g_xx, w = fHessian(x)
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
            print(f(z00))
            #print(z00)
            
        if stopCriterion == 0:
            if (L2Norm2(g_x, dimN) < atol) :  # stop when meet the relative and absolute tolerances
                break
        elif stopCriterion == 2:
            if abs(f(z00)-f(zmin))<errtol:
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
        print("function value =", f(z00))
    print("norm of gradient=", math.sqrt(L2Norm2(fDer(z00),dimN)))
    if printHessianEigenvalues==True:
        tmp, w=fHessian(z00)
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
        delta = lr_pure[-1]
        #    delta_tmp = delta
        x = z00
        f_x = f(x)
        g_x = fDer(x)
        gx_norm = L2Norm2(g_x, dimN)
        g_xx, w=fHessian(x)
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
                check = f(x_check) - f_x + alpha * delta * gx_norm
        elif check < 0:
            while check < 0 and delta <= delta0:
                count += 1
                delta = delta / beta  # rescale delta
                x_check = x - delta * g_x
                check = f(x - delta * g_x) - f_x + alpha * delta * gx_norm
            delta = delta * beta

        lr_pure.append(delta)
        #productGraph.append(delta*gxx_norm)
        
        gn = delta * fDer(z00)
        z00 = z00 - gn
        if verbose:
            print(f(z00))
            #print(z00)
            
        if stopCriterion == 0:
            if (L2Norm2(g_x, dimN) < atol) :  # stop when meet the relative and absolute tolerances
                break
        elif stopCriterion == 2:
            if abs(f(z00)-f(zmin))<errtol:
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
        print("function value =", f(z00))
    print("norm of gradient=", math.sqrt(L2Norm2(fDer(z00),dimN)))
    if printHessianEigenvalues==True:
        tmp, w=fHessian(z00)
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
        delta = lr_pure[-1]
        #    delta_tmp = delta
        x = z00
        f_x = f(x)
        g_x = fDer(x)
        gx_norm = L2Norm2(g_x, dimN)
        g_xx, w=fHessian(x)
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
                check = f(x_check) - f_x + alpha * delta * gx_norm
        elif check < 0:
            while check < 0 and delta <= delta0:
                count += 1
                delta = delta / beta  # rescale delta
                x_check = x - delta * g_x
                check = f(x - delta * g_x) - f_x + alpha * delta * gx_norm
            delta = delta * beta

        lr_pure.append(delta)
        #productGraph.append(delta*gxx_norm)
        
        gn = delta * fDer(z00)
        
        kkapp=1
        while constraintChect(z00-kkapp*gn)==False:
            kkapp=kkapp/2
            
    
        z00 = z00 - kkapp*gn
        
        if verbose:
            print(f(z00))
            #print(z00)
            
        if stopCriterion == 0:
            if (L2Norm2(g_x, dimN) < atol) :  # stop when meet the relative and absolute tolerances
                break
        elif stopCriterion == 2:
            if abs(f(z00)-f(zmin))<errtol:
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
        print("function value =", f(z00))
    print("norm of gradient=", math.sqrt(L2Norm2(fDer(z00),dimN)))
    if printHessianEigenvalues==True:
        tmp, w=fHessian(z00)
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
        delta = lr_pure[-1]
        #    delta_tmp = delta
        x = z00
        f_x = f(x)
        g_x = fDer(x)
        gx_norm = L2Norm2(g_x, dimN)
        gx_normSquareRoot=math.sqrt(gx_norm)
        g_xx, w=fHessian(x)
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
                check = f(x_check) - f_x + alpha * delta * gx_norm
        elif check < 0:
            while check < 0 and delta <= UnboundedLR(gx_normSquareRoot,1e+100, delta0):
                count += 1
                delta = delta / beta  # rescale delta
                x_check = x - delta * g_x
                check = f(x - delta * g_x) - f_x + alpha * delta * gx_norm
            delta = delta * beta

        lr_pure.append(delta)
        #productGraph.append(delta*gxx_norm)
        
        gn = delta * fDer(z00)
        z00 = z00 - gn
        if verbose:
            print(f(z00))
            #print(z00)
            
        if stopCriterion == 0:
            if (L2Norm2(g_x, dimN) < atol) :  # stop when meet the relative and absolute tolerances
                break
        elif stopCriterion == 2:
            if abs(f(z00)-f(zmin))<errtol:
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
        print("function value =", f(z00))
    
    #print(fDer(z00))
    flag=0
    for i in range(len(lr_pure)):
        if delta0<lr_pure[i]:
            flag+=1
    print("Number of learning rates bigger than \delta _0=", flag)
    print("norm of gradient=", math.sqrt(L2Norm2(fDer(z00),dimN)))
    if printHessianEigenvalues==True:
        tmp, w=fHessian(z00)
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
        delta = lr_pure[-1]
        #    delta_tmp = delta
        x = z00
        f_x = f(x)
        g_x = fDer(x)
        gx_norm = L2Norm2(g_x, dimN)
        gx_normSquareRoot=math.sqrt(gx_norm)
        g_xx, w=fHessian(x)
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
                check = f(x_check) - f_x + alpha * delta * gx_norm
        elif check < 0:
            while check < 0 and delta <= UnboundedLR(gx_normSquareRoot,1e+100, delta0):
                count += 1
                delta = delta / beta  # rescale delta
                x_check = x - delta * g_x
                check = f(x - delta * g_x) - f_x + alpha * delta * gx_norm
            delta = delta * beta

        lr_pure.append(delta)
        #productGraph.append(delta*gxx_norm)
        
        gn = delta * fDer(z00)
        
        kkapp=1
        while constraintChect(z00-kkapp*gn)==False:
            kkapp=kkapp/2
            
    
        z00 = z00 - kkapp*gn
        
        if verbose:
            print(f(z00))
            #print(z00)
            
        if stopCriterion == 0:
            if (L2Norm2(g_x, dimN) < atol) :  # stop when meet the relative and absolute tolerances
                break
        elif stopCriterion == 2:
            if abs(f(z00)-f(zmin))<errtol:
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
        print("function value =", f(z00))
    
    #print(fDer(z00))
    flag=0
    for i in range(len(lr_pure)):
        if delta0<lr_pure[i]:
            flag+=1
    print("Number of learning rates bigger than \delta _0=", flag)
    print("norm of gradient=", math.sqrt(L2Norm2(fDer(z00),dimN)))
    if printHessianEigenvalues==True:
        tmp, w=fHessian(z00)
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

        gn = fDer(z00)

        gamman=(m+1)**(-0.5)

    

        z00 = z00 + gamman * (lambda1*z00-(1/beta1)*psin-beta1*gn)

        psin=psin+gamman*(lambda1*z00-(1/beta1)*psin)
        if verbose:
            print(f(z00))
            #print(z00)
            
        if stopCriterion == 0:
            if (L2Norm2(gn, dimN) < atol) :  # stop when meet the relative and absolute tolerances
                break
        elif stopCriterion == 2:
            if abs(f(z00)-f(zmin))<errtol:
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
        print("function value =", f(z00))
    print("norm of gradient=", math.sqrt(L2Norm2(fDer(z00),dimN)))
    if printHessianEigenvalues==True:
        tmp, w=fHessian(z00)
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

        gn = fDer(z00)

        gamman=(m+1)**(-0.5)
        
        ggn=gamman * (lambda1*z00-(1/beta1)*psin-beta1*gn)
        
        kkapp=1
        while constraintChect(z00+kkapp*ggn)==False:
            kkapp=kkapp/2
            
    
        z00 = z00 + kkapp*ggn

        

        psin=psin+gamman*(lambda1*z00-(1/beta1)*psin)
        if verbose:
            print(f(z00))
            #print(z00)
            
        if stopCriterion == 0:
            if (L2Norm2(gn, dimN) < atol) :  # stop when meet the relative and absolute tolerances
                break
        elif stopCriterion == 2:
            if abs(f(z00)-f(zmin))<errtol:
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
        print("function value =", f(z00))
    print("norm of gradient=", math.sqrt(L2Norm2(fDer(z00),dimN)))
    if printHessianEigenvalues==True:
        tmp, w=fHessian(z00)
        #print(fDer(z00_old))
        print("Eigenvalues of Hessian=", w)
    return
    

### Functions V_1, V_2
def V1(theta):
    tmp=(1/4)*(1-math.cos(theta))
    return tmp

def V2(r,xiOne,xiTwo):
    
    Cxi=(1/8)*(1+xiOne+xiTwo+5*xiOne*xiTwo)
    tmp=4*((r**(-12))-Cxi*(r**(-6)))
    return tmp


### protein folding with n=3
def f46a(z):
    #z=theta_2
    #V1=(1/4)*(1-math.cos(z))
    r13= np.sqrt((math.cos(z))**2 + (math.sin(z))**2)
    #C13= (1/8)*(1+xi1+xi3 + 5*xi1*xi3)
    #V2 = 4*(r13**(-12)-C13*(r13**(-6)))
    tmp=V1(z)+V2(r13,xi1,xi3)
    return tmp

def f46aDer(z):
    
    f46aG=nd.Gradient(f46a)
    tmp=f46aG(z)
    return tmp

def f46aHessian(z):
    f46aH=nd.Gradient(f46aDer)
    tmp=f46aH(z)
    
    return tmp, tmp

def f46aConstraint(z):
    tmp=True
    tmp=(z>-math.pi) and (z<math.pi)
    return tmp

def f46aInitialization():
    z=np.random.uniform(-1,1)*math.pi
    xi1= random.choice([-1, 1])
    xi3= random.choice([-1, 1])
    
    tmpZ=z
    tmpXi1=xi1
    tmpXi3=xi3
    
    return tmpZ, tmpXi1,tmpXi3

### protein folding with n=4
def f46b(z):
    #z=theta_2, theta_3
    z2=z[0] # theta_2
    z3=z[1] # theta_3
    #V1=(1/4)*(1-math.cos(z2)) + (1/4)*(1-math.cos(z3))
    r13 = np.sqrt((math.cos(z2))**2 + (math.sin(z2))**2)
    r14 = np.sqrt((math.cos(z2)+math.cos(z2+z3))**2 + (math.sin(z2)+math.sin(z2+z3))**2)
    r24 = np.sqrt((math.cos(z3))**2 + (math.sin(z3))**2)
    
    #C13= (1/8)*(1+xi1+xi3 + 5*xi1*xi3)
    #C24= (1/8)*(1+xi2+xi4 + 5*xi2*xi4)
    
    #V2 = 4*(r13**(-12)-C13*r13**(-6)) + 4*(r24**(-12)-C24*r24**(-6))
    tmp=V1(z2)+V1(z3)+V2(r13,xi1,xi3)+V2(r14,xi1,xi4)+V2(r24,xi2,xi4)
    return tmp

def f46bDer(z):
    
    f46bG=nd.Gradient(f46b)
    tmp=f46bG(z)
    return tmp

def f46bHessian(z):
    f46bH=nd.Hessian(f46b)
    tmp=f46bH(z)
    w,v=LA.eig(tmp)
    return tmp, w

def f46bConstraint(z):
    tmp=True
    tmp=(z[0]>-math.pi) and (z[0]<math.pi) and (z[1]>-math.pi) and ((z[1]<math.pi))
    return tmp

def f46bInitialization():
    z=[np.random.uniform(-1,1)*math.pi,np.random.uniform(-1,1)*math.pi]
    xi1= random.choice([-1, 1])
    xi3= random.choice([-1, 1])
    xi2= random.choice([-1, 1])
    xi4= random.choice([-1, 1])
    
    tmpZ=z
    tmpXi1=xi1
    tmpXi2=xi2
    tmpXi3=xi3
    tmpXi4=xi4
    
    return tmpZ, tmpXi1,tmpXi2,tmpXi3,tmpXi4

### protein folding with n=5
def f46c(z):
    #z=theta_2, theta_3
    z2=z[0] # theta_2
    z3=z[1] # theta_3
    z4=z[2] # theta_4
    #V1=(1/4)*(1-math.cos(z2)) + (1/4)*(1-math.cos(z3))
    r13 = np.sqrt((math.cos(z2))**2 + (math.sin(z2))**2)
    r14 = np.sqrt((math.cos(z2)+math.cos(z2+z3))**2 + (math.sin(z2)+math.sin(z2+z3))**2)
    r15 = np.sqrt((math.cos(z2)+math.cos(z2+z3)+math.cos(z2+z3+z4))**2 + (math.sin(z2)+math.sin(z2+z3)+math.sin(z2+z3+z4))**2)
    r24 = np.sqrt((math.cos(z3))**2 + (math.sin(z3))**2)
    r25 = np.sqrt((math.cos(z3)+math.cos(z3+z4))**2 + (math.sin(z3)+math.sin(z3+z4))**2)
    r35 = np.sqrt((math.cos(z4))**2 + (math.sin(z4))**2)
    
    #C13= (1/8)*(1+xi1+xi3 + 5*xi1*xi3)
    #C24= (1/8)*(1+xi2+xi4 + 5*xi2*xi4)
    
    #V2 = 4*(r13**(-12)-C13*r13**(-6)) + 4*(r24**(-12)-C24*r24**(-6))
    tmp=V1(z2)+V1(z3)++V1(z4)+V2(r13,xi1,xi3)+V2(r14,xi1,xi4)+V2(r15,xi1,xi5)+V2(r24,xi2,xi4)+V2(r25,xi2,xi5)+V2(r35,xi3,xi5)
    return tmp

def f46cDer(z):
    
    f46cG=nd.Gradient(f46c)
    tmp=f46cG(z)
    return tmp

def f46cHessian(z):
    f46cH=nd.Hessian(f46c)
    tmp=f46cH(z)
    w,v=LA.eig(tmp)
    return tmp, w

### protein folding with n=6
def f46d(z):
    #z=theta_2, theta_3
    z2=z[0] # theta_2
    z3=z[1] # theta_3
    z4=z[2] # theta_4
    z5=z[3] # theta_5
    
    #V1=(1/4)*(1-math.cos(z2)) + (1/4)*(1-math.cos(z3))
    r13 = np.sqrt((math.cos(z2))**2 + (math.sin(z2))**2)
    r14 = np.sqrt((math.cos(z2)+math.cos(z2+z3))**2 + (math.sin(z2)+math.sin(z2+z3))**2)
    r15 = np.sqrt((math.cos(z2)+math.cos(z2+z3)+math.cos(z2+z3+z4))**2 + (math.sin(z2)+math.sin(z2+z3)+math.sin(z2+z3+z4))**2)
    r16 = np.sqrt((math.cos(z2)+math.cos(z2+z3)+math.cos(z2+z3+z4)+math.cos(z2+z3+z4+z5))**2 + (math.sin(z2)+math.sin(z2+z3)+math.sin(z2+z3+z4)+math.sin(z2+z3+z4+z5))**2)
    
    r24 = np.sqrt((math.cos(z3))**2 + (math.sin(z3))**2)
    r25 = np.sqrt((math.cos(z3)+math.cos(z3+z4))**2 + (math.sin(z3)+math.sin(z3+z4))**2)
    r26 = np.sqrt((math.cos(z3)+math.cos(z3+z4)+math.cos(z3+z4+z5))**2 + (math.sin(z3)+math.sin(z3+z4)+math.sin(z3+z4+z5))**2)
    
    
    r35 = np.sqrt((math.cos(z4))**2 + (math.sin(z4))**2)
    r36 = np.sqrt((math.cos(z4)+math.cos(z4+z5))**2 + (math.sin(z4)+math.sin(z4+z5))**2)
    
    r46 = np.sqrt((math.cos(z5))**2 + (math.sin(z5))**2)
    
    
    
    #C13= (1/8)*(1+xi1+xi3 + 5*xi1*xi3)
    #C24= (1/8)*(1+xi2+xi4 + 5*xi2*xi4)
    
    #V2 = 4*(r13**(-12)-C13*r13**(-6)) + 4*(r24**(-12)-C24*r24**(-6))
    tmp=V1(z2)+V1(z3)+V1(z4)+V1(z5)+V2(r13,xi1,xi3)+V2(r14,xi1,xi4)+V2(r15,xi1,xi5)+V2(r16,xi1,xi6)+V2(r24,xi2,xi4)+V2(r25,xi2,xi5)+V2(r26,xi2,xi6)+V2(r35,xi3,xi5)+V2(r36,xi3,xi6)+V2(r46,xi4,xi6)
    return tmp

def f46dDer(z):
    
    f46dG=nd.Gradient(f46d)
    tmp=f46dG(z)
    return tmp

def f46dHessian(z):
    f46dH=nd.Hessian(f46d)
    tmp=f46dH(z)
    w,v=LA.eig(tmp)
    return tmp, w

### protein folding with n=7
def f46e(z):
    #z=theta_2, theta_3
    z2=z[0] # theta_2
    z3=z[1] # theta_3
    z4=z[2] # theta_4
    z5=z[3] # theta_5
    z6=z[4] # theta_6
    
    #V1=(1/4)*(1-math.cos(z2)) + (1/4)*(1-math.cos(z3))
    r13 = np.sqrt((math.cos(z2))**2 + (math.sin(z2))**2)
    r14 = np.sqrt((math.cos(z2)+math.cos(z2+z3))**2 + (math.sin(z2)+math.sin(z2+z3))**2)
    r15 = np.sqrt((math.cos(z2)+math.cos(z2+z3)+math.cos(z2+z3+z4))**2 + (math.sin(z2)+math.sin(z2+z3)+math.sin(z2+z3+z4))**2)
    r16 = np.sqrt((math.cos(z2)+math.cos(z2+z3)+math.cos(z2+z3+z4)+math.cos(z2+z3+z4+z5))**2 + (math.sin(z2)+math.sin(z2+z3)+math.sin(z2+z3+z4)+math.sin(z2+z3+z4+z5))**2)
    r17 = np.sqrt((math.cos(z2)+math.cos(z2+z3)+math.cos(z2+z3+z4)+math.cos(z2+z3+z4+z5)+math.cos(z2+z3+z4+z5+z6))**2 + (math.sin(z2)+math.sin(z2+z3)+math.sin(z2+z3+z4)+math.sin(z2+z3+z4+z5)+math.sin(z2+z3+z4+z5+z6))**2)

    
    r24 = np.sqrt((math.cos(z3))**2 + (math.sin(z3))**2)
    r25 = np.sqrt((math.cos(z3)+math.cos(z3+z4))**2 + (math.sin(z3)+math.sin(z3+z4))**2)
    r26 = np.sqrt((math.cos(z3)+math.cos(z3+z4)+math.cos(z3+z4+z5))**2 + (math.sin(z3)+math.sin(z3+z4)+math.sin(z3+z4+z5))**2)
    r27 = np.sqrt((math.cos(z3)+math.cos(z3+z4)+math.cos(z3+z4+z5)+math.cos(z3+z4+z5+z6))**2 + (math.sin(z3)+math.sin(z3+z4)+math.sin(z3+z4+z5)+math.sin(z3+z4+z5+z6))**2)
    
    
    r35 = np.sqrt((math.cos(z4))**2 + (math.sin(z4))**2)
    r36 = np.sqrt((math.cos(z4)+math.cos(z4+z5))**2 + (math.sin(z4)+math.sin(z4+z5))**2)
    r37 = np.sqrt((math.cos(z4)+math.cos(z4+z5)+math.cos(z4+z5+z6))**2 + (math.sin(z4)+math.sin(z4+z5)+math.sin(z4+z5+z6))**2)
    
    
    r46 = np.sqrt((math.cos(z5))**2 + (math.sin(z5))**2)
    r47 = np.sqrt((math.cos(z5)+math.cos(z5+z6))**2 + (math.sin(z5)+math.sin(z5+z6))**2)
    
    r57 = np.sqrt((math.cos(z6))**2 + (math.sin(z6))**2)
    
    
    #C13= (1/8)*(1+xi1+xi3 + 5*xi1*xi3)
    #C24= (1/8)*(1+xi2+xi4 + 5*xi2*xi4)
    
    #V2 = 4*(r13**(-12)-C13*r13**(-6)) + 4*(r24**(-12)-C24*r24**(-6))
    tmp=V1(z2)+V1(z3)+V1(z4)+V1(z5)+V1(z6)+V2(r13,xi1,xi3)+V2(r14,xi1,xi4)+V2(r15,xi1,xi5)+V2(r16,xi1,xi6)+V2(r17,xi1,xi7)+V2(r24,xi2,xi4)+V2(r25,xi2,xi5)+V2(r26,xi2,xi6)+V2(r27,xi2,xi7)+V2(r35,xi3,xi5)+V2(r36,xi3,xi6)+V2(r37,xi3,xi7)+V2(r46,xi4,xi6)+V2(r47,xi4,xi7)+V2(r57,xi5,xi7)
    return tmp

def f46eDer(z):
    
    f46eG=nd.Gradient(f46e)
    tmp=f46eG(z)
    return tmp

def f46eHessian(z):
    f46eH=nd.Hessian(f46e)
    tmp=f46eH(z)
    w,v=LA.eig(tmp)
    return tmp, w


### protein folding with n=10
def f46f(z):
    #z=theta_2, theta_3
    z2=z[0] # theta_2
    z3=z[1] # theta_3
    z4=z[2] # theta_4
    z5=z[3] # theta_5
    z6=z[4] # theta_6
    z7=z[5] # theta_7
    z8=z[6] # theta_8
    z9=z[7] # theta_9
    
    #V1=(1/4)*(1-math.cos(z2)) + (1/4)*(1-math.cos(z3))
    r13 = np.sqrt((math.cos(z2))**2 + (math.sin(z2))**2)
    r14 = np.sqrt((math.cos(z2)+math.cos(z2+z3))**2 + (math.sin(z2)+math.sin(z2+z3))**2)
    r15 = np.sqrt((math.cos(z2)+math.cos(z2+z3)+math.cos(z2+z3+z4))**2 + (math.sin(z2)+math.sin(z2+z3)+math.sin(z2+z3+z4))**2)
    r16 = np.sqrt((math.cos(z2)+math.cos(z2+z3)+math.cos(z2+z3+z4)+math.cos(z2+z3+z4+z5))**2 + (math.sin(z2)+math.sin(z2+z3)+math.sin(z2+z3+z4)+math.sin(z2+z3+z4+z5))**2)
    r17 = np.sqrt((math.cos(z2)+math.cos(z2+z3)+math.cos(z2+z3+z4)+math.cos(z2+z3+z4+z5)+math.cos(z2+z3+z4+z5+z6))**2 + (math.sin(z2)+math.sin(z2+z3)+math.sin(z2+z3+z4)+math.sin(z2+z3+z4+z5)+math.sin(z2+z3+z4+z5+z6))**2)
    r18 = np.sqrt((math.cos(z2)+math.cos(z2+z3)+math.cos(z2+z3+z4)+math.cos(z2+z3+z4+z5)+math.cos(z2+z3+z4+z5+z6)+math.cos(z2+z3+z4+z5+z6+z7))**2 + (math.sin(z2)+math.sin(z2+z3)+math.sin(z2+z3+z4)+math.sin(z2+z3+z4+z5)+math.sin(z2+z3+z4+z5+z6)+math.sin(z2+z3+z4+z5+z6+z7))**2)
    r19 = np.sqrt((math.cos(z2)+math.cos(z2+z3)+math.cos(z2+z3+z4)+math.cos(z2+z3+z4+z5)+math.cos(z2+z3+z4+z5+z6)+math.cos(z2+z3+z4+z5+z6+z7)+math.cos(z2+z3+z4+z5+z6)+math.cos(z2+z3+z4+z5+z6+z7+z8))**2 + (math.sin(z2)+math.sin(z2+z3)+math.sin(z2+z3+z4)+math.sin(z2+z3+z4+z5)+math.sin(z2+z3+z4+z5+z6)+math.sin(z2+z3+z4+z5+z6+z7)+math.sin(z2+z3+z4+z5+z6+z7+z8))**2)
    r10 = np.sqrt((math.cos(z2)+math.cos(z2+z3)+math.cos(z2+z3+z4)+math.cos(z2+z3+z4+z5)+math.cos(z2+z3+z4+z5+z6)+math.cos(z2+z3+z4+z5+z6+z7)+math.cos(z2+z3+z4+z5+z6)+math.cos(z2+z3+z4+z5+z6+z7+z8)+math.cos(z2+z3+z4+z5+z6+z7+z8+z9))**2 + (math.sin(z2)+math.sin(z2+z3)+math.sin(z2+z3+z4)+math.sin(z2+z3+z4+z5)+math.sin(z2+z3+z4+z5+z6)+math.sin(z2+z3+z4+z5+z6+z7)+math.sin(z2+z3+z4+z5+z6+z7+z8)+math.sin(z2+z3+z4+z5+z6+z7+z8+z9))**2)



    r24 = np.sqrt((math.cos(z3))**2 + (math.sin(z3))**2)
    r25 = np.sqrt((math.cos(z3)+math.cos(z3+z4))**2 + (math.sin(z3)+math.sin(z3+z4))**2)
    r26 = np.sqrt((math.cos(z3)+math.cos(z3+z4)+math.cos(z3+z4+z5))**2 + (math.sin(z3)+math.sin(z3+z4)+math.sin(z3+z4+z5))**2)
    r27 = np.sqrt((math.cos(z3)+math.cos(z3+z4)+math.cos(z3+z4+z5)+math.cos(z3+z4+z5+z6))**2 + (math.sin(z3)+math.sin(z3+z4)+math.sin(z3+z4+z5)+math.sin(z3+z4+z5+z6))**2)
    r28 = np.sqrt((math.cos(z3)+math.cos(z3+z4)+math.cos(z3+z4+z5)+math.cos(z3+z4+z5+z6)+math.cos(z3+z4+z5+z6+z7))**2 + (math.sin(z3)+math.sin(z3+z4)+math.sin(z3+z4+z5)+math.sin(z3+z4+z5+z6)+math.sin(z3+z4+z5+z6+z7))**2)
    r29 = np.sqrt((math.cos(z3)+math.cos(z3+z4)+math.cos(z3+z4+z5)+math.cos(z3+z4+z5+z6)+math.cos(z3+z4+z5+z6+z7)+math.cos(z3+z4+z5+z6+z7+z8))**2 + (math.sin(z3)+math.sin(z3+z4)+math.sin(z3+z4+z5)+math.sin(z3+z4+z5+z6)+math.sin(z3+z4+z5+z6+z7)+math.sin(z3+z4+z5+z6+z7+z8))**2)
    
    r20 = np.sqrt((math.cos(z3)+math.cos(z3+z4)+math.cos(z3+z4+z5)+math.cos(z3+z4+z5+z6)+math.cos(z3+z4+z5+z6+z7)+math.cos(z3+z4+z5+z6+z7+z8)+math.cos(z3+z4+z5+z6+z7+z8+z9))**2 + (math.sin(z3)+math.sin(z3+z4)+math.sin(z3+z4+z5)+math.sin(z3+z4+z5+z6)+math.sin(z3+z4+z5+z6+z7)+math.sin(z3+z4+z5+z6+z7+z8)+math.sin(z3+z4+z5+z6+z7+z8+z9))**2)
    
    r35 = np.sqrt((math.cos(z4))**2 + (math.sin(z4))**2)
    r36 = np.sqrt((math.cos(z4)+math.cos(z4+z5))**2 + (math.sin(z4)+math.sin(z4+z5))**2)
    r37 = np.sqrt((math.cos(z4)+math.cos(z4+z5)+math.cos(z4+z5+z6))**2 + (math.sin(z4)+math.sin(z4+z5)+math.sin(z4+z5+z6))**2)
    r38 = np.sqrt((math.cos(z4)+math.cos(z4+z5)+math.cos(z4+z5+z6)+math.cos(z4+z5+z6+z7))**2 + (math.sin(z4)+math.sin(z4+z5)+math.sin(z4+z5+z6)+math.sin(z4+z5+z6+z7))**2)
    r39 = np.sqrt((math.cos(z4)+math.cos(z4+z5)+math.cos(z4+z5+z6)+math.cos(z4+z5+z6+z7)+math.cos(z4+z5+z6+z7+z8))**2 + (math.sin(z4)+math.sin(z4+z5)+math.sin(z4+z5+z6)+math.sin(z4+z5+z6+z7)+math.sin(z4+z5+z6+z7+z8))**2)
    r30 = np.sqrt((math.cos(z4)+math.cos(z4+z5)+math.cos(z4+z5+z6)+math.cos(z4+z5+z6+z7)+math.cos(z4+z5+z6+z7+z8)+math.cos(z4+z5+z6+z7+z8+z9))**2 + (math.sin(z4)+math.sin(z4+z5)+math.sin(z4+z5+z6)+math.sin(z4+z5+z6+z7)+math.sin(z4+z5+z6+z7+z8)+math.sin(z4+z5+z6+z7+z8+z9))**2)
    
    r46 = np.sqrt((math.cos(z5))**2 + (math.sin(z5))**2)
    r47 = np.sqrt((math.cos(z5)+math.cos(z5+z6))**2 + (math.sin(z5)+math.sin(z5+z6))**2)
    r48 = np.sqrt((math.cos(z5)+math.cos(z5+z6)+math.cos(z5+z6+z7))**2 + (math.sin(z5)+math.sin(z5+z6)+math.sin(z5+z6+z7))**2)
    r49 = np.sqrt((math.cos(z5)+math.cos(z5+z6)+math.cos(z5+z6+z7)+math.cos(z5+z6+z7+z8))**2 + (math.sin(z5)+math.sin(z5+z6)+math.sin(z5+z6+z7)+math.sin(z5+z6+z7+z8))**2)
    r40 = np.sqrt((math.cos(z5)+math.cos(z5+z6)+math.cos(z5+z6+z7)+math.cos(z5+z6+z7+z8)+math.cos(z5+z6+z7+z8+z9))**2 + (math.sin(z5)+math.sin(z5+z6)+math.sin(z5+z6+z7)+math.sin(z5+z6+z7+z8)+math.sin(z5+z6+z7+z8+z9))**2)
    
    r57 = np.sqrt((math.cos(z6))**2 + (math.sin(z6))**2)
    r58 = np.sqrt((math.cos(z6)+math.cos(z6+z7))**2 + (math.sin(z6)+math.sin(z6+z7))**2)
    r59 = np.sqrt((math.cos(z6)+math.cos(z6+z7)+math.cos(z6+z7+z8))**2 + (math.sin(z6)+math.sin(z6+z7)+math.sin(z6+z7+z8))**2)
    r50 = np.sqrt((math.cos(z6)+math.cos(z6+z7)+math.cos(z6+z7+z8)+math.cos(z6+z7+z8+z9))**2 + (math.sin(z6)+math.sin(z6+z7)+math.sin(z6+z7+z8)+math.sin(z6+z7+z8+z9))**2)
    
    r68 = np.sqrt((math.cos(z7))**2 + (math.sin(z7))**2)
    r69 = np.sqrt((math.cos(z7)+math.cos(z7+z8))**2 + (math.sin(z7)+math.sin(z7+z8))**2)
    r60 = np.sqrt((math.cos(z7)+math.cos(z7+z8)+math.cos(z7+z8+z9))**2 + (math.sin(z7)+math.sin(z7+z8)+math.sin(z7+z8+z9))**2)
    
    r79 = np.sqrt((math.cos(z8))**2 + (math.sin(z8))**2)
    r70 = np.sqrt((math.cos(z8)+math.cos(z8+z9))**2 + (math.sin(z8)+math.sin(z8+z9))**2)
    
    r80 = np.sqrt((math.cos(z9))**2 + (math.sin(z9))**2)
    
    
    
    #C13= (1/8)*(1+xi1+xi3 + 5*xi1*xi3)
    #C24= (1/8)*(1+xi2+xi4 + 5*xi2*xi4)
    
    #V2 = 4*(r13**(-12)-C13*r13**(-6)) + 4*(r24**(-12)-C24*r24**(-6))
    tmp=V1(z2)+V1(z3)+V1(z4)+V1(z5)+V1(z6)+V1(z7)+V1(z8)+V1(z9)+V2(r13,xi1,xi3)+V2(r14,xi1,xi4)+V2(r15,xi1,xi5)+V2(r16,xi1,xi6)+V2(r17,xi1,xi7)+V2(r18,xi1,xi8)+V2(r19,xi1,xi9)+V2(r10,xi1,xi10)+V2(r24,xi2,xi4)+V2(r25,xi2,xi5)+V2(r26,xi2,xi6)+V2(r27,xi2,xi7)+V2(r28,xi2,xi8)+V2(r29,xi2,xi9)+V2(r20,xi2,xi10)+V2(r35,xi3,xi5)+V2(r36,xi3,xi6)+V2(r37,xi3,xi7)+V2(r38,xi3,xi8)+V2(r39,xi3,xi9)+V2(r30,xi3,xi10)+V2(r46,xi4,xi6)+V2(r47,xi4,xi7)+V2(r48,xi4,xi8)+V2(r49,xi4,xi9)+V2(r40,xi4,xi10)+V2(r57,xi5,xi7)+V2(r58,xi5,xi8)+V2(r59,xi5,xi9)+V2(r50,xi5,xi10)+V2(r68,xi6,xi8)+V2(r69,xi6,xi9)+V2(r60,xi6,xi10)+V2(r79,xi7,xi9)+V2(r70,xi7,xi10)+V2(r80,xi8,xi10)
    return tmp

def f46fDer(z):
    
    f46fG=nd.Gradient(f46f)
    tmp=f46fG(z)
    return tmp

def f46fHessian(z):
    f46fH=nd.Hessian(f46f)
    tmp=f46fH(z)
    w,v=LA.eig(tmp)
    return tmp, w

### protein folding with n=7
def f46e(z):
    #z=theta_2, theta_3
    z2=z[0] # theta_2
    z3=z[1] # theta_3
    z4=z[2] # theta_4
    z5=z[3] # theta_5
    z6=z[4] # theta_6
    
    #V1=(1/4)*(1-math.cos(z2)) + (1/4)*(1-math.cos(z3))
    r13 = np.sqrt((math.cos(z2))**2 + (math.sin(z2))**2)
    r14 = np.sqrt((math.cos(z2)+math.cos(z2+z3))**2 + (math.sin(z2)+math.sin(z2+z3))**2)
    r15 = np.sqrt((math.cos(z2)+math.cos(z2+z3)+math.cos(z2+z3+z4))**2 + (math.sin(z2)+math.sin(z2+z3)+math.sin(z2+z3+z4))**2)
    r16 = np.sqrt((math.cos(z2)+math.cos(z2+z3)+math.cos(z2+z3+z4)+math.cos(z2+z3+z4+z5))**2 + (math.sin(z2)+math.sin(z2+z3)+math.sin(z2+z3+z4)+math.sin(z2+z3+z4+z5))**2)
    r17 = np.sqrt((math.cos(z2)+math.cos(z2+z3)+math.cos(z2+z3+z4)+math.cos(z2+z3+z4+z5)+math.cos(z2+z3+z4+z5+z6))**2 + (math.sin(z2)+math.sin(z2+z3)+math.sin(z2+z3+z4)+math.sin(z2+z3+z4+z5)+math.sin(z2+z3+z4+z5+z6))**2)

    
    r24 = np.sqrt((math.cos(z3))**2 + (math.sin(z3))**2)
    r25 = np.sqrt((math.cos(z3)+math.cos(z3+z4))**2 + (math.sin(z3)+math.sin(z3+z4))**2)
    r26 = np.sqrt((math.cos(z3)+math.cos(z3+z4)+math.cos(z3+z4+z5))**2 + (math.sin(z3)+math.sin(z3+z4)+math.sin(z3+z4+z5))**2)
    r27 = np.sqrt((math.cos(z3)+math.cos(z3+z4)+math.cos(z3+z4+z5)+math.cos(z3+z4+z5+z6))**2 + (math.sin(z3)+math.sin(z3+z4)+math.sin(z3+z4+z5)+math.sin(z3+z4+z5+z6))**2)
    
    
    r35 = np.sqrt((math.cos(z4))**2 + (math.sin(z4))**2)
    r36 = np.sqrt((math.cos(z4)+math.cos(z4+z5))**2 + (math.sin(z4)+math.sin(z4+z5))**2)
    r37 = np.sqrt((math.cos(z4)+math.cos(z4+z5)+math.cos(z4+z5+z6))**2 + (math.sin(z4)+math.sin(z4+z5)+math.sin(z4+z5+z6))**2)
    
    
    r46 = np.sqrt((math.cos(z5))**2 + (math.sin(z5))**2)
    r47 = np.sqrt((math.cos(z5)+math.cos(z5+z6))**2 + (math.sin(z5)+math.sin(z5+z6))**2)
    
    r57 = np.sqrt((math.cos(z6))**2 + (math.sin(z6))**2)
    
    
    #C13= (1/8)*(1+xi1+xi3 + 5*xi1*xi3)
    #C24= (1/8)*(1+xi2+xi4 + 5*xi2*xi4)
    
    #V2 = 4*(r13**(-12)-C13*r13**(-6)) + 4*(r24**(-12)-C24*r24**(-6))
    tmp=V1(z2)+V1(z3)+V1(z4)+V1(z5)+V1(z6)+V2(r13,xi1,xi3)+V2(r14,xi1,xi4)+V2(r15,xi1,xi5)+V2(r16,xi1,xi6)+V2(r17,xi1,xi7)+V2(r24,xi2,xi4)+V2(r25,xi2,xi5)+V2(r26,xi2,xi6)+V2(r27,xi2,xi7)+V2(r35,xi3,xi5)+V2(r36,xi3,xi6)+V2(r37,xi3,xi7)+V2(r46,xi4,xi6)+V2(r47,xi4,xi7)+V2(r57,xi5,xi7)
    return tmp

def f46eDer(z):
    
    f46eG=nd.Gradient(f46e)
    tmp=f46eG(z)
    return tmp

def f46eHessian(z):
    f46eH=nd.Hessian(f46e)
    tmp=f46eH(z)
    w,v=LA.eig(tmp)
    return tmp, w




###### Test some values in the paper

##### Test function f46a

#z=0.61866*math.pi
#xi1=1
#xi2=1
#xi3=1
#print("function value=", f46a(z))
    
z=0*math.pi
xi1=1
xi2=1
xi3=1
print("function value=", f46a(z))

#z=0.61866*math.pi
#xi1=1
#xi2=-1
#xi3=1
#print("function value=", f46a(z))

#z=0*math.pi
#xi1=-1
#xi2=1
#xi3=-1
#print("function value=", f46a(z))


#### Test function f46b

#z=np.array(0*math.pi)
xi1=-1
xi2=-1
xi3=-1
xi4=-1
xi5=-1
xi6=-1
xi7=1
xi8=-1
xi9=1
xi10=-1
#print("function value=", f46a(z))




################  Run algorithms


D=3
mm=1

dimN=D


bound=5

ggam=1

printLastPoint=True

printHessianEigenvalues=True

verbose=False
stopCriterion=0
#stopCriterion=1
#stopCriterion=2

NIterate=5000
#NIterate=10000

print("atol=", atol)
aatol=1e-20

#z00_old=np.array([1.07689387,2.97081771,0.800213082])
z00_old=np.array([np.random.uniform(-1,1)*math.pi for _ in range(dimN)])
#z00_old=np.array([10 for _ in range(dimN)])

rrtol=1e-10

print("the function is", "f46c")
f=f46c
fDer=f46cDer
fHessian=f46cHessian

print("Number of iterates=", NIterate)

print("initial point=", z00_old)

tmp, w=fHessian(z00_old)
print("derivative at the initial point=", fDer(z00_old))
print("Hessian at the initial point=", tmp)
print("Eigenvalues of the Hessian at the initial point=", w)
print("function value at initial point=", f(z00_old))


z00InPaper=np.array([0.34345*math.pi, 0.56501*math.pi, 0.09318*math.pi])
print("Optimal value according to the paper=", f(z00InPaper))

#NewtonMethod()
#RandomNewQNewton()

#RandomNewtonMethod()
#NewQNewton()
#BFGS()
#InertialNewtonM()

#BacktrackingGD()
#AdaptiveCubicRegularisation()
#UnboundedTwoWayBacktrackingGD()
#TwoWayBacktrackingGD()
