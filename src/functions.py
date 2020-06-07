import math
import numpy as np
from numpy import linalg as LA
import params as pr
import scipy
import scipy.special as scisp
from scipy.optimize import fmin_bfgs

"""Here list all cost functions reported in experiments. 
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


######### Function f1 ##############
### We look at the function f(t)=t^{1+gamma }, where 0<gamma <1.


def f1(t):
    tmp = abs(t) ** (1 + gamma)
    return tmp


def f1Der(t):
    if (t > 0):
        tmp = (1 + gamma) * (abs(t) ** gamma)
    else:
        tmp = -(1 + gamma) * (abs(t) ** gamma)
    return tmp


def f1Hessian(t):
    
    tmp = gamma * (1 + gamma) * (abs(t) ** (gamma - 1))

    return tmp, tmp


########## Function f2 ############################
### We look at the function f(t)=|t|^(gamma), where 0< gamma <1


def f2(t):
    tmp = abs(t) ** (gamma)
    return tmp


def f2Der(t):
    if (t > 0):
        tmp = gamma * (abs(t) ** (gamma - 1))
    else:
        tmp = -gamma * (abs(t) ** (gamma - 1))
   
    return tmp


def f2Hessian(t):
    
    tmp = gamma * (gamma - 1) * (abs(t) ** (gamma - 2))
    
    return tmp, tmp
    
    
#print(f2Hessian(0.5))


########## Function f3 ########################
### We look at the function f3(x)=e^{-1/x^2}

def f3(t):
    if (t == 0):
        tmp = 0
    else:
        tmp = math.exp(-1 / (t * t))
    return tmp


def f3Der(t):
    if (t == 0):
        tmp = 0
    else:
        tmp = 2 * (1 / (t * t * t)) * (math.exp(-1 / (t * t)))
    return tmp


def f3Hessian(t):
    if (t == 0):
        tmp = 0
    else:
        tmp = -6 * (1 / (t * t * t * t)) * (math.exp(-1 / (t * t))) + 4 * (1 / (t * t * t * t * t * t)) * (
            math.exp(-1 / (t * t)))
    return tmp, tmp


########### Function f4 #######################
### We look at the function f4(t)=t^3 sin (1/t)

def f4(t):
    if (t == 0):
        tmp = 0
    else:
        tmp = t * t * t * np.sin(1 / t)
    return tmp


def f4Der(t):
    if (t == 0):
        tmp = 0
    else:
        tmp = 3 * t * t * np.sin(1 / t) - t * np.cos(1 / t)
    return tmp


def f4Hessian(t):
    if (t == 0):
        tmp = 1
    else:
        tmp = 6 * t * np.sin(1 / t) - (1 / t) * np.sin(1 / t) - 4 * np.cos(1 / t)

    return tmp, tmp


######### Function f5 ################################
### We look at the function f5(t)=t^3 cos(1/t)


def f5(t):
    if (t == 0):
        tmp = 0
    else:
        tmp = t * t * t * np.cos(1 / t)
    return tmp


def f5Der(t):
    if (t == 0):
        tmp = 0
    else:
        tmp = 3 * t * t * np.cos(1 / t) + t * np.sin(1 / t)
    return tmp


def f5Hessian(t):
    if (t == 0):
        tmp = 1
    else:
        tmp = 4 * np.sin(1 / t) + 6 * t * np.cos(1 / t) - (1 / t) * np.cos(1 / t)

    return tmp, tmp


######### Function f6 ###############################
### We look at the function f6=x^3 sin(1/x) + c6*y^3 sin (1/y)

def f6(z):
    x, y = z

    tmp = f4(x) + c6 * f4(y)
    return tmp


def f6Der(z):
    x, y = z

    tmp = np.array([f4Der(x), c6 * f4Der(y)])
    return tmp


def f6Hessian(z):
    x, y = z

    ux, ux1 = f4Hessian(x)
    uy, uy1 = f4Hessian(y)

    tmp = np.array([[ux, 0], [0, c6 * uy]])
    w, v = LA.eig(tmp)
    return tmp, w


######### Function f7 #######################
### We look at the function f7 = (x-1)^2+100(y-x^2)^2
### The behaviour is the same as for the function f7'=0.01 f7.


def f7(z):
    x, y = z
    tmp = (a - x) * (a - x) + b * (y - x * x) * (y - x * x)
    #tmp = (1/b)*tmp
    return tmp


def f7Der(z):
    x, y = z
    tmp = np.array([2 * (x - a) + 4 * b * x * (x * x - y), 2 * b * (y - x * x)])
    #tmp = (1/b)*tmp
    return tmp


def f7Hessian(z):
    x, y = z
    tmp = np.array([[2 + 4 * b * (x * x - y) + 8 * b * (x * x), -4 * b * x], [-4 * b * x, 2 * b]])
    #tmp = (1/b)*tmp
    w, v = LA.eig(tmp)
    return tmp, w


############ Function f8 #######################################
### We look at the function f8=e^(t^2)-2t^3


def f8(t):
    tmp = math.exp(t * t) - 2 * (t * t * t)
    return tmp


def f8Der(t):
    tmp = 2 * t * math.exp(t * t) - 6 * (t * t)
    return tmp


def f8Hessian(t):
    tmp = 2 * math.exp(t * t) + 4 * (t * t) * math.exp(t * t) - 12 * t
    return tmp, tmp


########## Function f9 ################################
### We look at the function f9=x^3 sin (1/x) - y^3 sin(1/y)

def f9(z):
    x, y = z

    tmp = f4(x) + c9 * f4(y)
    return tmp


def f9Der(z):
    x, y = z

    tmp = np.array([f4Der(x), c9 * f4Der(y)])
    return tmp


def f9Hessian(z):
    x, y = z

    ux, ux1 = f4Hessian(x)
    uy, uy1 = f4Hessian(y)

    tmp = np.array([[ux, 0], [0, c9 * uy]])
    w, v = LA.eig(tmp)
    return tmp, w


######### Function f10 ################################
### We look at the function f10= f7(x1,x2)+f7(x2,x3)+f7(x3,x4)

def f10(z):
    u, v, t, s = z
    tmp = ((a - u) * (a - u) + b * (v - u * u) * (v - u * u)) + ((a - v) * (a - v) + b * (t - v * v) * (t - v * v)) + (
            (a - t) * (a - t) + b * (s - t * t) * (s - t * t))
    return tmp


def f10Der(z):
    u, v, t, s = z
    tmp = np.array(
        [-2 * (a - 2 * b * u * u * u + 2 * b * u * v - u), -2 * (a - v + b * (2 * t * v + u * u - 2 * v * v * v - v)),
         -2 * (a + t * (2 * b * s - b - 1) - 2 * b * t * t * t + b * v * v), 2 * b * (s - t * t)])
    return tmp


def f10Hessian(z):
    u, v, t, s = z
    tmp = np.array([[12 * b * u * u - 4 * b * v + 2, -4 * b * u, 0, 0],
                    [-4 * b * u, 2 * (-2 * b * t + 6 * b * v * v + b + 1), -4 * b * v, 0],
                    [0, -4 * b * v, 2 * (-2 * b * s + 6 * b * t * t + b + 1), -4 * b * t], [0, 0, -4 * b * t, 2 * b]])
    w1, w2 = LA.eig(tmp)
    return tmp, w1


########## Function f11 ##############################
####### We look at the function f11(t)=PReLU.

def f11(t):
    # If coef=0, then we have the ReLU function
    # If coef<>0, then we have the PReLU function.
    # If coef = 0.01, then we have the Leaky ReLU function
    # If coef =-1, then we have the absolute function |t|.
    # Because the second derivative of f11 is 0, we put it to be a non-zero small number ep, so that Newton's method or Local Backtracking GD can run.

    if (t < 0):
        tmp = coef * t
    else:
        tmp = t
    return tmp


def f11Der(t):
    if (t < 0):
        tmp = coef
    else:
        tmp = 1
    return tmp


def f11Hessian(t):
    tmp = 0
    w = 0
    return tmp, w


########## Function f12 ##############################
####### We look at the function f12(x,y,s)=5*PReLU(x)+sign1*PReLU(y)+sign2*s. Here we will choose (sign1 ,sign2) to be either (\pm 1,0) or (0,\pm 1)


def f12(z):
    x, y, s = z

    tmp = 5 * f11(x) + sign1 * f11(y) + sign2 * s
    return tmp


def f12Der(z):
    x, y,s = z
    tmp = np.array([5 * f11Der(x), sign1 * f11Der(y), sign2])
    return tmp


def f12Hessian(z):
    tmp = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    w, v = LA.eig(tmp)
    return tmp, w


######### Function f13 #################################
######  We look at the function f(x,y) = 100(y-|x|)^2 + |1-x|

def f13(z):
    x, y = z

    tmp = 100*(y-abs(x))*(y-abs(x))+abs(1-x)
    return tmp


def f13Der(z):
    x, y = z
    if (x<0):
        tmp=np.array([200*(x+y)-1,200*x+200*y])
    elif (x<1):
        tmp = np.array([200*(x-y)-1,200*(y-x)])
    else:
        tmp = np.array([200*(x-y)+1,200*(y-x)])
    
    
    return tmp


def f13Hessian(z):
    x, y = z
    if (x<0):
        tmp=np.array([[200,200],[200,200]])
    elif (x<1):
        tmp=np.array([[200,-200],[-200,200]])
    else:
        tmp=np.array([[200,-200],[-200,200]])
    w, v = LA.eig(tmp)
    return tmp, w


##################################################
#print(f13Der(np.array([0.30703369, 0.30953365]))+f13Der(np.array([0.50952063, 0.50705168]))+f13Der(np.array([1.50938185, 1.50688189]))+f13Der(np.array([1.30689492, 1.30936386])))

print(f13Der(np.array([1.00255712, 1.00006335]))+f13Der(np.array([6.33544128e-05, 2.55711999e-03])))


######## Function f14 ############################
### We look at the function f(x,y) = 5 |x|+y

def f14(z):
    x, y = z

    tmp = 5*abs(x)+y
    return tmp


def f14Der(z):
    x, y = z
    if (x<0):
        tmp=np.array([-5,1])
    else:
        tmp = np.array([5,1])
    
    
    return tmp


def f14Hessian(z):
    x, y = z
    
    tmp=np.array([[0,0],[0,0]])
    w, v = LA.eig(tmp)
    return tmp, w

########### Function f15 #######################
### We look at the function f15(t)=t^5 sin (1/t)

def f15(t):
    if (t == 0):
        tmp = 0
    else:
        tmp = t * t * t *t*t* np.sin(1 / t)
    return tmp


def f15Der(t):
    if (t == 0):
        tmp = 0
    else:
        tmp = 5 * t * t *t*t* np.sin(1 / t) - t *t*t* np.cos(1 / t)
    return tmp


def f15Hessian(t):
    if (t == 0):
        tmp = 0
    else:
        tmp = 20 * t*t*t * np.sin(1 / t) - t * np.sin(1 / t) - 8*t*t * np.cos(1 / t)

    return tmp, tmp


########### Function f16 #######################
### We look at the function f16(t)=(t^4/4)-(2t^3/3)-(11t/2)+12t. From the Wikipedia page for Newton's method.

def f16(t):
    
    tmp = (t*t*t*t/4)-(2*t*t*t/3)-(11*t*t/2)+(12*t)
    return tmp


def f16Der(t):
    
    tmp = (t*t*t)-(2*t*t)-(11*t)+12
    return tmp


def f16Hessian(t):
    
    tmp = (3*t*t)-(4*t)-11

    return tmp, tmp

########### Function f17 #######################
### We look at the function f17(t)=(t^4/4)-t^2+2t. From the Wikipedia page for Newton's method.

def f17(t):
    
    tmp = (t*t*t*t/4)-(t*t)+(2*t)
    return tmp


def f17Der(t):
    
    tmp = (t*t*t)-(2*t)+2
    return tmp


def f17Hessian(t):
    
    tmp = (3*t*t)-2

    return tmp, tmp
    
########### Function f18 #######################
### We look at the function f18(t)=4/3 ci(2/t)+t(t^2-2)sin(2/t)/3+t^2/2+t^2cos(2/t)/3. From the Wikipedia page for Newton's method.

def f18(t):
    
    tmp = (4*scisp.sici(2/t)[1]/3)+(t*(t*t-2)*math.sin(2/t)/3)+(t*t/2)+(t*t*math.cos(2/t)/3)
    return tmp


def f18Der(t):
    
    tmp = t+(t*t*math.sin(2/t))
    return tmp


def f18Hessian(t):
    
    tmp = 1+(2*t*math.sin(2/t))-(2*math.cos(2/t))

    return tmp, tmp
    
######### Function f19 ########################
### We look at the function f(x,y)=x^2+y^2+4xy. Here to check the advantage of the orthogonal version of Signed correction of Newton's method

def f19(z):
    x, y = z

    tmp = x*x+y*y+4*x*y
    return tmp


def f19Der(z):
    x, y = z
    
    tmp = np.array([2*x+4*y,2*y+4*x])
    
    
    return tmp


def f19Hessian(z):
    x, y = z
    
    tmp=np.array([[2,4],[4,2]])
    w, v = LA.eig(tmp)
    return tmp, w

######### Function f20 ########################
### We look at the function f(x,y)=x^2+y^2+xy. Here to check the advantage of the orthogonal version of Signed correction of Newton's method

def f20(z):
    x, y = z

    tmp = x*x+y*y+x*y
    return tmp


def f20Der(z):
    x, y = z
    
    tmp = np.array([2*x+y,2*y+x])
    
    
    return tmp


def f20Hessian(z):
    x, y = z
    
    tmp=np.array([[2,1],[1,2]])
    w, v = LA.eig(tmp)
    return tmp, w


######### Function f21 ########################
### We look at the function f(x,y)=x^2+y^2+2xy. Here to check the advantage of the orthogonal version of Signed correction of Newton's method

def f21(z):
    x, y = z

    tmp = x*x+y*y+2*x*y
    return tmp


def f21Der(z):
    x, y = z
    
    tmp = np.array([2*x+2*y,2*y+2*x])
    
    
    return tmp


def f21Hessian(z):
    x, y = z
    delta =1
    tmp=np.array([[2,2],[2,2]])
    w, v = LA.eig(tmp)
    return tmp, w

######### Function f22 ########################
### We look at the function f(x,y,t) a quadratic, with eigenvalues 0,1,-1. Here to check the advantage of the orthogonal version of Signed correction of Newton's method

def f22(z):
    x, y,t = z

    tmp = (-184*x*x-488*x*y+320*x*t)+(-488*y*x-316*y*y+1240*y*t)+(320*t*x+1240*t*y-400*t*t)
    return tmp/(2*8)


def f22Der(z):
    x, y,t = z
    
    tmp = np.array([-184*x-488*y+320*t,-488*x-316*y+1240*t,320*x+1240*y-400*t])
    
    
    return tmp/8


def f22Hessian(z):
    x, y,t = z
    
    tmp=np.array([[-184,-488,320],[-488,-316,1240],[320,1240,-400]])/8
    w, v = LA.eig(tmp)
    return tmp, w


#Check BFGS, already available on python
#x0 = pr.z0f2


#xopt = fmin_bfgs(f2, x0,fprime=f2Der,maxiter=pr.NIterate,gtol=pr.rtol)
#print(xopt)

#print(f1Hessian(xopt))

