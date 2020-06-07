import numpy as np
import math
import time
from numpy import linalg as LA
from random import random
from utils import L2Norm2, CheckCriticalType, UnboundedLR,  NegativeOrthogonalDecomposition, cutoff
import params as pr
import matplotlib.pyplot as plt
from random import *
import scipy.special as scisp
from scipy.optimize import fmin_bfgs

"""Detailed Definition of Newton's method, Random damping Newton's method, New Q-Newton's method, Inertial Newton's method and BFGS
"""
gamma0 = pr.gamma0
choiceAdam = pr.choiceAdam








def NewtonMethod(f, fDer, fHessian, z0, NIterate, mode, gamma0, v0, stopCriterion, rtol, atol, dimN, verbose):
    if mode == 'gd':
        print("Use Newton method")
    else:
        print("Use Newton method for NAG with gamma ", gamma0, "; initial direction =", v0)

    print("With parameters: f=", f, "; initial point =", z0)
    print("We will stop when both z_{n+1}-z_n and f(z_{n+1})-f(z_n) are small")
    zn = z0

    t0 = time.time()

    print("Number of maximum iterates:", NIterate, "; But note that the number of actual iterates may be much smaller")
    for m in range(NIterate):
        tmp, w = fHessian(zn)
        if dimN == 1:
            HessTr = tmp
            HessInv = 1 / HessTr
            gn = fDer(zn) * HessInv
        elif dimN > 1:
            HessTr = tmp.transpose()
            HessInv = LA.inv(HessTr)
            gn = np.matmul(fDer(zn), HessInv)
        zn_old = zn
        zn = zn - gn
        if verbose:
            print("Step = ", m, "; zn=", zn_old, "; Square of Displacement = ||z_{n+1}-z_n||^2= ", L2Norm2(gn, dimN),
                  " ; Square of gradient ", L2Norm2(fDer(zn), dimN))

        if stopCriterion == 0:
            if (L2Norm2(gn, dimN) < atol) and (
                    abs(f(zn) - f(zn_old)) < rtol):  # stop when meet the relative and absolute tolerances
                break
        elif stopCriterion == 1:
            if (abs(f(zn)) < rtol):  # stop when meet the relative and absolute tolerances
                break
    print("Time:", time.time() - t0, "; Step:", m, "; z_n:", zn, "; f(z_n):", f(zn), ": Error:",
          L2Norm2(fDer(zn), dimN))
    CheckCriticalType(fHessian, zn)

    return




def NewQNewtonMethod(f, fDer, fHessian, z0, NIterate, mode, gamma0, v0, stopCriterion, rtol, atol, dimN, verbose):
    if mode == 'gd':
        print("Use New Q-Newton method with pertubation by orthogonal matrices Full version")
    else:
        print("Use New Q-Newton method with pertubation by orthogonal matrices Full version for NAG with gamma ", gamma0, "; initial direction =", v0)

    
    print("With parameters: f=", f, "; initial point =", z0)
    print("We will stop when both z_{n+1}-z_n and f(z_{n+1})-f(z_n) are small")
    
    zn = z0

    t0 = time.time()
    
    delta =1
    

    print("Number of maximum iterates:", NIterate, "; But note that the number of actual iterates may be much smaller")
    for m in range(NIterate):
        tmp, w = fHessian(zn)
        if dimN == 1:
            HessTr = tmp
            if HessTr==0:
                HessTr=HessTr+delta*cutoff(L2Norm2(fDer(zn),dimN))
            
            HessInv = 1 / HessTr
            gn = fDer(zn) * HessInv
            gn=NegativeOrthogonalDecomposition(HessTr,gn,dimN)
        elif dimN > 1:
            HessTr = tmp.transpose()
            if LA.det(HessTr)==0:
                HessTr=HessTr+delta*cutoff(L2Norm2(fDer(zn),dimN))*np.identity(dimN,dtype=float)
            HessInv = LA.inv(HessTr)
            gn = np.matmul(fDer(zn), HessInv)
            gn=NegativeOrthogonalDecomposition(HessTr,gn,dimN)
        zn_old = zn
        zn = zn - gn
        if verbose:
            print("Step = ", m, "; zn=", zn_old, "; Square of Displacement = ||z_{n+1}-z_n||^2= ", L2Norm2(gn, dimN),
                  " ; Square of gradient ", L2Norm2(fDer(zn), dimN))

        if stopCriterion == 0:
            if (L2Norm2(gn, dimN) < atol) and (
                    abs(f(zn) - f(zn_old)) < rtol):  # stop when meet the relative and absolute tolerances
                break
        elif stopCriterion == 1:
            if (abs(f(zn)) < rtol):  # stop when meet the relative and absolute tolerances
                break
    print("Time:", time.time() - t0, "; Step:", m, "; z_n:", zn, "; f(z_n):", f(zn), ": Error:",
          L2Norm2(fDer(zn), dimN))
    CheckCriticalType(fHessian, zn)

    return



def RandomNewtonMethod(f, fDer, fHessian, z0, NIterate, mode, gamma0, v0,
                       stopCriterion, rtol, atol, dimN, verbose):
    if mode == 'gd':
        print("Use Random Newton method")
    else:
        print("Use Random Newton method for NAG with gamma ", gamma0, "; initial direction =", v0)

    minLR = 0
    maxLR = 2
    print("With parameters: f=", f, "; initial point =", z0)
    print("We will stop when both z_{n+1}-z_n and f(z_{n+1})-f(z_n) are small")
    zn = z0

    t0 = time.time()
    lr_pure = []
    print("Number of maximum iterates:", NIterate, "; But note that the number of actual iterates may be much smaller")
    for m in range(NIterate):

        delta = minLR + random() * (maxLR - minLR)
        lr_pure.append(delta)
        tmp, w = fHessian(zn)
        if dimN == 1:
            HessTr = tmp
            HessInv = 1 / HessTr
            gn = delta * fDer(zn) * HessInv
        elif dimN > 1:
            HessTr = tmp.transpose()
            HessInv = LA.inv(HessTr)
            gn = delta * np.matmul(fDer(zn), HessInv)
        zn_old = zn
        zn = zn - gn
        if verbose:
            print("Step = ", m, "; zn=", zn_old, "; Square of Displacement = ||z_{n+1}-z_n||^2= ", L2Norm2(gn, dimN),
                  " ; Square of gradient ", L2Norm2(fDer(zn), dimN))

        if stopCriterion == 0:
            if (L2Norm2(gn, dimN) < atol) and L2Norm2(fDer(zn), dimN) < atol and (
                    abs(f(zn) - f(zn_old)) < rtol):  # stop when meet the relative and absolute tolerances
                break;
        elif stopCriterion == 1:
            if (abs(f(zn)) < rtol):  # stop when meet the relative and absolute tolerances
                break;
    print("Time:", time.time() - t0, "; Step:", m, "; z_n:", zn, "; f(z_n):", f(zn), ": Error:",
          L2Norm2(fDer(zn), dimN))
    CheckCriticalType(fHessian, zn)

    return




def InertiaNewtonMethod(f, fDer, fHessian, z0, delta0, NIterate, mode, gamma0, v0,
           stopCriterion, rtol, atol, dimN, verbose):
    if mode == 'gd':
        print("Use Inertia Newton method")
    else:
        print("Use Inertia Newton method for NAG with gamma ", gamma0, "; initial direction =", v0)

    alpha1 = 0.5

    beta1 = 0.1

    lambda1 = (1/beta1) -alpha1

    print("With parameters: f=", f, "; initial point =", z0)

    print("We will stop when both z_{n+1}-z_n and f(z_{n+1})-f(z_n) are small")

    zn = z0

    psin=v0



    t0 = time.time()

    print("Number of maximum iterates:", NIterate, "; But note that the number of actual iterates may be much smaller")

    for m in range(NIterate):

        gn = fDer(zn)
    
        gamman=(m+1)**(-0.5)

        zn_old = zn

        zn = zn + gamman * (lambda1*zn-(1/beta1)*psin-beta1*gn)
    
        psin=psin+gamman*(lambda1*zn-(1/beta1)*psin)

        if verbose:
            print("Step = ", m, "; zn=", zn_old, "; Square of Displacement = ||z_{n+1}-z_n||^2= ", L2Norm2(gn, dimN),
              " ; Square of gradient ", L2Norm2(fDer(zn), dimN))

        if stopCriterion == 0:

            if (L2Norm2(gn, dimN) < atol) and (
                abs(f(zn) - f(zn_old)) < rtol):  # stop when meet the relative and absolute tolerances

                break;

        elif stopCriterion == 1:

            if (abs(f(zn)) < rtol):  # stop when meet the relative and absolute tolerances

                break;

    print("Time:", time.time() - t0, "; Step:", m, "; z_n:", zn, "; f(z_n):", f(zn), ": Error:", L2Norm2(gn, dimN))

    CheckCriticalType(fHessian, zn)

    return
    
    
def BFGS(f, fDer, fHessian, z0, delta0, NIterate, mode, gamma0, v0,
           stopCriterion, rtol, atol, dimN, verbose):
    if mode == 'gd':
        print("Use BFGS method")
    else:
        print("Use BFGS method for NAG with gamma ", gamma0, "; initial direction =", v0)

    xopt=fmin_bfgs(f, z0,fprime=fDer,maxiter=pr.NIterate,gtol=pr.rtol)
    print(xopt)
    
    return

    
    
    
