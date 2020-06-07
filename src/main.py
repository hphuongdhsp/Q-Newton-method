#Authors: Tuan Hang Nguyen and Tuyen Trung Truong


from functions import *
from Experiment import Experiment
import params as pr


"""Doing experiments using: Newton's method, BFGS, New Q-Newton's method, Random damping Newton's method and Inertial Newton's method """

### List of functions

# 1D functions experiments
expr1d = {
    'f1': {'main': f1, 'derivative': f1Der, 'hessian': f1Hessian,
           'params': {'dim': 1, 'z0': pr.z0f1, 'v0': pr.v0D1}},
    'f2': {'main': f2, 'derivative': f2Der, 'hessian': f2Hessian,
           'params': {'dim': 1, 'z0': pr.z0f2, 'v0': pr.v0D1}},
    'f3': {'main': f3, 'derivative': f3Der, 'hessian': f3Hessian,
           'params': {'dim': 1, 'z0': pr.z0f3, 'v0': pr.v0D1}},
    'f4': {'main': f4, 'derivative': f4Der, 'hessian': f4Hessian,
           'params': {'dim': 1, 'z0': pr.z0f4, 'v0': pr.v0D1}},
    'f5': {'main': f5, 'derivative': f5Der, 'hessian': f5Hessian,
           'params': {'dim': 1, 'z0': pr.z0f5, 'v0': pr.v0D1}},
    'f8C1': {'main': f8, 'derivative': f8Der, 'hessian': f8Hessian,
             'params': {'dim': 1, 'z0': pr.z0f8C1, 'v0': pr.v0D1}},
    'f8C2': {'main': f8, 'derivative': f8Der, 'hessian': f8Hessian,
             'params': {'dim': 1, 'z0': pr.z0f8C2, 'v0': pr.v0D1}},
    'f8C3': {'main': f8, 'derivative': f8Der, 'hessian': f8Hessian,
             'params': {'dim': 1, 'z0': pr.z0f8C3, 'v0': pr.v0D1}},
    'f11': {'main': f11, 'derivative': f11Der, 'hessian': f11Hessian,
            'params': {'dim': 1, 'z0': pr.z0f11, 'v0': pr.v0D1}},
    'f15': {'main': f15, 'derivative': f15Der, 'hessian': f15Hessian,
    'params': {'dim': 1, 'z0': pr.z0f15, 'v0': pr.v0D1}},
    'f16C1': {'main': f16, 'derivative': f16Der, 'hessian': f16Hessian,
    'params': {'dim': 1, 'z0': pr.z0f16C1, 'v0': pr.v0D1}},
    'f16C2': {'main': f16, 'derivative': f16Der, 'hessian': f16Hessian,
    'params': {'dim': 1, 'z0': pr.z0f16C2, 'v0': pr.v0D1}},
    'f16C3': {'main': f16, 'derivative': f16Der, 'hessian': f16Hessian,
    'params': {'dim': 1, 'z0': pr.z0f16C3, 'v0': pr.v0D1}},
    'f16C4': {'main': f16, 'derivative': f16Der, 'hessian': f16Hessian,
    'params': {'dim': 1, 'z0': pr.z0f16C4, 'v0': pr.v0D1}},
    'f16C5': {'main': f16, 'derivative': f16Der, 'hessian': f16Hessian,
    'params': {'dim': 1, 'z0': pr.z0f16C5, 'v0': pr.v0D1}},
    'f17': {'main': f17, 'derivative': f17Der, 'hessian': f17Hessian,
    'params': {'dim': 1, 'z0': pr.z0f17, 'v0': pr.v0D1}},
    'f18': {'main': f18, 'derivative': f18Der, 'hessian': f18Hessian,
    'params': {'dim': 1, 'z0': pr.z0f18, 'v0': pr.v0D1}},
}

# 2D functions experiments
expr2d = {
    'f6': {'main': f6, 'derivative': f6Der, 'hessian': f6Hessian,
           'params': {'dim': 2, 'z0': pr.z0f6, 'v0': pr.v0D2}},
    'f7': {'main': f7, 'derivative': f7Der, 'hessian': f7Hessian,
           'params': {'dim': 2, 'z0': pr.z0f7, 'v0': pr.v0D2}},
    'f9': {'main': f9, 'derivative': f9Der, 'hessian': f9Hessian,
           'params': {'dim': 2, 'z0': pr.z0f9, 'v0': pr.v0D2}},
    'f13': {'main': f13, 'derivative': f13Der, 'hessian': f13Hessian,
    'params': {'dim': 2, 'z0': pr.z0f13, 'v0': pr.v0D2}},
    'f14': {'main': f14, 'derivative': f14Der, 'hessian': f14Hessian,
    'params': {'dim': 2, 'z0': pr.z0f14, 'v0': pr.v0D2}},
    'f19': {'main': f19, 'derivative': f19Der, 'hessian': f19Hessian,
    'params': {'dim': 2, 'z0': pr.z0f19, 'v0': pr.v0D2}},
    'f20': {'main': f20, 'derivative': f20Der, 'hessian': f20Hessian,
    'params': {'dim': 2, 'z0': pr.z0f20, 'v0': pr.v0D2}},
    'f21': {'main': f21, 'derivative': f21Der, 'hessian': f21Hessian,
       'params': {'dim': 2, 'z0': pr.z0f21, 'v0': pr.v0D2}},
}

# 3D functions experiments
expr3d = {
    'f12': {'main': f12, 'derivative': f12Der, 'hessian': f12Hessian,
            'params': {'dim': 3, 'z0': pr.z0f12, 'v0': pr.v0D3}},
        'f22': {'main': f22, 'derivative': f22Der, 'hessian': f22Hessian,
            'params': {'dim': 3, 'z0': pr.z0f22, 'v0': pr.v0D3}},
}

# 4D functions experiments
expr4d = {
    'f10': {'main': f10, 'derivative': f10Der, 'hessian': f10Hessian,
            'params': {'dim': 4, 'z0': pr.z0f10, 'v0': pr.v0D4}}
}

### All cost functions
all_expr = {}
for f in [expr1d, expr2d, expr3d, expr4d]:
    all_expr.update(f)

### List of experiments to be done
# Please see detailed initial points of each experiment in params.py
exprList = [
     'f1',  # f(t)=t^{1+gamma }, where 0<gamma <1.
     #'f2',  # f(t)=|t|^(gamma), where 0< gamma <1
     #'f3',  # f3(x)=e^{-1/x^2}
     #'f4',  # f4(t)=t^3 sin (1/t)
     #'f5',  # f5(t)=t^3 cos(1/t)
     #'f8C1',  # f8=e^(t^2)-2t^3
    #'f8C2',  # f8=e^(t^2)-2t^3
    # 'f8C3',  # f8=e^(t^2)-2t^3
    # 'f11',  # f11(t)=PReLU
    # 'f6',  # f6=x^3 sin(1/x) + c6*y^3 sin (1/y)
    # 'f7',  # f7 = (x-1)^2+100(y-x^2)^2
    # 'f9',  # f9=x^3 sin (1/x) - y^3 sin(1/y)
    # 'f12',  # f12(x,y,s)=5*PReLU(x)+sign1*PReLU(y)+sign2*s. (sign1 ,sign2) to be either (\pm 1,0) or (0,\pm 1)
    #'f10',  # f10= f7(x1,x2)+f7(x2,x3)+f7(x3,x4)
    #'f13',  # f13= 100(y-|x|)^2+|1-x|
    #'f14',  # f14= 5|x|+y
    #'f15',  # f15= t^5 sin(1/t)
    #'f16C1',  # f16= (t^4/4)-(2t^3/3)-(11t/2)+12t
    #'f16C2',  # f16= (t^4/4)-(2t^3/3)-(11t/2)+12t
    #'f16C3',  # f16= (t^4/4)-(2t^3/3)-(11t/2)+12t
    #'f16C4',  # f16= (t^4/4)-(2t^3/3)-(11t/2)+12t
    #'f16C5',  # f16= (t^4/4)-(2t^3/3)-(11t/2)+12t
    #'f17',  # f17= (t^4/4)-t^2+2t
    #'f18',  # f18= 4/3 ci(2/t)+t(t^2-2)sin(2/t)/3+t^2/2+t^2cos(2/t)/3
    # 'f19',  # f19= x^2+y^2+4xy
    # 'f20',  # f20= x^2+y^2+xy
     #'f21',  # f21= x^2+y^2+2xy
    #'f22',# quadratic in 3 variables, with eigenvalues 0,1,-1
]

####### Normal GD or NAG
mode = 'gd'  # 'gd': normal Gradient Descent, 'nag': Nesterov Momentum
#mode = 'nag'  # 'gd': normal Gradient Descent, 'nag': Nesterov Momentum


####### Expriments
for f in exprList:
    print('Experiments on function', f)
    dimN = all_expr[f]['params']['dim']
    z0 = all_expr[f]['params']['z0']
    v0 = all_expr[f]['params']['v0']
    with np.errstate(all="raise"):
        Experiment(all_expr[f], z0, pr.delta0, pr.alpha, pr.beta, pr.delta0N, pr.NIterate, mode, pr.gamma0, v0,
                   pr.stopCriterion, pr.rtol, pr.atol,
                   dimN, pr.n1, pr.n2, pr.p, pr.deltaMin, pr.deltaMax, pr.nn, pr.verbose)
