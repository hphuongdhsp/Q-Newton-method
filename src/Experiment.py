import numpy as np
from functions import *
from methods import *
import params as pr

"""Experiment: Apply the assigned gradient descent methods"""

METHODS = pr.METHODS

def Experiment(func, z0, delta0, alpha, beta, delta0N, NIterate, mode, gamma0, v0, stopCriterion, rtol, atol, dimN,
               n1, n2, p, deltaMin, deltaMax, nn, verbose):
    f, fDer, fHessian = func['main'], func['derivative'], func['hessian']

    # with np.errstate(all="raise"):

    for i, method in enumerate(METHODS):
        print(i, '-', method)

        try:
            # if True:
            if method == "NEWTON METHOD":

                NewtonMethod(f, fDer, fHessian, z0, NIterate, mode, gamma0, v0, stopCriterion, rtol, atol, dimN, verbose)

            elif method == "NEW Q NEWTON METHOD":

                NewQNewtonMethod(f, fDer, fHessian, z0, NIterate, mode, gamma0, v0, stopCriterion, rtol, atol, dimN, verbose)

            elif method == "RANDOM NEWTON METHOD":

                RandomNewtonMethod(f, fDer, fHessian, z0, NIterate, mode, gamma0, v0,
                stopCriterion, rtol, atol, dimN, verbose)

            elif method == "INERTIAL NEWTON METHOD":

                InertiaNewtonMethod(f, fDer, fHessian, z0, delta0, NIterate, mode, gamma0, v0,
                stopCriterion, rtol, atol, dimN, verbose)

            elif method == "BFGS":

                BFGS(f, fDer, fHessian, z0, delta0, NIterate, mode, gamma0, v0,
                stopCriterion, rtol, atol, dimN, verbose)

            
            
            else:
                print('Wrong method name!')

        except Exception as ex:
            print("Exception raised: {}".format(ex))
            print("The method diverged")
