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
atol=pr.atol
rtol=pr.rtol
#D=pr.D

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
### We look at the function f(x,y)=x^2+y^2+4xy.

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
### We look at the function f(x,y)=x^2+y^2+xy.

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
### We look at the function f(x,y)=x^2+y^2+2xy.

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
### We look at the function f(x,y,t) a quadratic, with eigenvalues 0,1,-1.

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

######### Function f23 ########################
### We look at the function f(x,y)=5x+|y|.

def f23(z):
    x, y = z

    tmp = 5*x+abs(y)
    return tmp


def f23Der(z):
    f23G=nd.Gradient(f23)(z)
    tmp=f23G
    
    return tmp


#def f23Der(z):
#    x,y=z
#    if y>0:
#        tmp=np.array([5.0,1.0])
#    else:
#        tmp=np.array([5.0,-1.0])
#    return tmp





def f23Hessian(z):
    f23H=nd.Hessian(f23)(z)
    tmp=f23H
    w, v = LA.eig(tmp)
    return tmp, w

#def f23Hessian(z):
#    x,y=z
#    tmp=np.array([[0.0,0.0],[0.0,0.0]])
#    w, v = LA.eig(tmp)
#    return tmp, w


########### f24: Ackley path function #################
######## Global minimum is at the origin. The variables are in the interval [-5,5].

def f24(z):
    
    
    tmp=-20*np.exp(-0.2*math.sqrt(np.sum(z**2)/D))-np.exp(np.sum(np.cos(z*2*math.pi))/D)+20+math.e
    return tmp


def f24Der(z):
    f24G=nd.Gradient(f24)
    tmp=f24G(z)
    return tmp

def f24Hessian(z):
    f24H=nd.Hessian(f24)
    tmp=f24H(z)
    w,v=LA.eig(tmp)
    return tmp, w

########### f25: Rastrigin function #################
######## Global minimum is at the origin. The variables are in the interval [-5.12,5.12].

def f25(z):
    A=10
    
    tmp=A*D+np.sum(z**2-A*np.cos(2*math.pi*z))
    return tmp


def f25Der(z):
    f25G=nd.Gradient(f25)
    tmp=f25G(z)
    return tmp

def f25Hessian(z):
    f25H=nd.Hessian(f25)
    tmp=f25H(z)
    w,v=LA.eig(tmp)
    return tmp, w

########### f26: Rosenbrock function #################
######## Global minimum is at the point (1,...,1). The variables are in the interval [-infinity,infinity].

def f26(z):
    onesvector=np.full(D,1,dtype=np.float128)
    zshift=np.full(D,0,dtype=np.float128)
    zsquare=np.full(D,0,dtype=np.float128)
    for i in range(D-1):
        zshift[i]=z[i+1]
    for i in range(D-1):
        zsquare[i]=z[i]*z[i]
    z1=z-onesvector
    tmp=np.sum(100*((zshift-zsquare)**2)+z1**2)
    return tmp


def f26Der(z):
    f26G=nd.Gradient(f26)
    tmp=f26G(z)
    return tmp

def f26Hessian(z):
    f26H=nd.Hessian(f26)
    tmp=f26H(z)
    w,v=LA.eig(tmp)
    return tmp, w


########### f27: Beale function #################
######## Global minimum is at the point (3,0.5). The variables are in the interval [-4.5,4.5].

def f27(z):
    x,y=z
    tmp=(1.5-x+x*y)*(1.5-x+x*y)+(2.25-x+x*y*y)*(2.25-x+x*y*y)+(2.625-x+x*y*y*y)*(2.625-x+x*y*y*y)
    return tmp


def f27Der(z):
    f27G=nd.Gradient(f27)
    tmp=f27G(z)
    return tmp

def f27Hessian(z):
    f27H=nd.Hessian(f27)
    tmp=f27H(z)
    w,v=LA.eig(tmp)
    return tmp, w



########### f28: Booth function #################
######## Global minimum is at the point (1,3). The variables are in the interval [-10,10].

def f28(z):
    x,y=z
    tmp=(x+2*y-7)*(x+2*y-7)+(2*x+y-5)*(2*x+y-5)
    return tmp


def f28Der(z):
    
    f28G=nd.Gradient(f28)
    tmp=f28G(z)
    return tmp

def f28Hessian(z):
    f28H=nd.Hessian(f28)
    tmp=f28H(z)
    w,v=LA.eig(tmp)
    return tmp, w


########### f29: Bukin function #6 #################
######## Global minimum is at the point (-10,1). The variables are in the interval [-10,10].

def f29(z):
    x,y=z
    tmp=100*math.sqrt(abs(y-0.01*x*x))+0.01*abs(x+10)
    return tmp


def f29Der(z):
    
    f29G=nd.Gradient(f29)
    tmp=f29G(z)
    return tmp

def f29Hessian(z):
    f29H=nd.Hessian(f29)
    tmp=f29H(z)
    w,v=LA.eig(tmp)
    return tmp, w

########### f30: Levi function #13 #################
######## Global minimum is at the point (1,1). The variables are in the interval [-10,10].

def f30(z):
    x,y=z
    tmp=np.sin(3*np.pi*x)*np.sin(3*np.pi*x)+(x-1)*(x-1)*(1+np.sin(3*np.pi*y)*np.sin(3*np.pi*y))+(y-1)*(y-1)*(1+np.sin(2*np.pi*y)*np.sin(2*np.pi*y))
    return tmp


def f30Der(z):
    
    f30G=nd.Gradient(f30)
    tmp=f30G(z)
    return tmp

def f30Hessian(z):
    f30H=nd.Hessian(f30)
    tmp=f30H(z)
    w,v=LA.eig(tmp)
    return tmp, w

########### f31: Himmelblau function #################
######## Global minimum are at the points (3,0), (-2.805118, 3.131312), (-3.779310, -3.283186) and (3.584428, -1.848126). The variables are in the interval [-10,10].

def f31(z):
    x,y=z
    tmp=(x*x+y-11)*(x*x+y-11)+(x+y*y-y)*(x+y*y-y)
    return tmp


def f31Der(z):
    
    f31G=nd.Gradient(f31)
    tmp=f31G(z)
    return tmp

def f31Hessian(z):
    f31H=nd.Hessian(f31)
    tmp=f31H(z)
    w,v=LA.eig(tmp)
    return tmp, w

########### f32: Easom function #################
######## Global minimum are at the points (\pi ,\pi ). The variables are in the interval [-10,10].

def f32(z):
    x,y=z
    tmp=-np.cos(x)*np.cos(y)*np.exp(-((x-np.pi)*(x-np.pi)+(y-np.pi)*(y-np.pi)))
    return tmp


def f32Der(z):
    
    f32G=nd.Gradient(f32)
    tmp=f32G(z)
    return tmp

def f32Hessian(z):
    f32H=nd.Hessian(f32)
    tmp=f32H(z)
    w,v=LA.eig(tmp)
    return tmp, w

########### f33: Cross in tray function #################
######## Global minimum are at 4 points (\pm 1.34941, \pm 1.34941).

def f33(z):
    x,y=z
    tmp=-0.0001*((1+abs(np.sin(x)*np.sin(y))*np.exp(abs(100-math.sqrt(x*x+y*y)/np.pi)))**0.1)
    return tmp


def f33Der(z):
    
    f33G=nd.Gradient(f33)
    tmp=f33G(z)
    return tmp

def f33Hessian(z):
    f33H=nd.Hessian(f33)
    tmp=f33H(z)
    w,v=LA.eig(tmp)
    return tmp, w


########### f34: Eggeholder  function #################
######## Global minimum are at (512,404.2319), the function value is $-959.6407$.

def f34(z):
    x,y=z
    tmp=-(y+47)*math.sin(math.sqrt(abs((x/2)+(y+47))))-x*math.sin(math.sqrt(abs((x)-(y+47))))
    return tmp


def f34Der(z):
    
    f34G=nd.Gradient(f34)
    tmp=f34G(z)
    return tmp

def f34Hessian(z):
    f34H=nd.Hessian(f34)
    tmp=f34H(z)
    w,v=LA.eig(tmp)
    return tmp, w

########### f35: McCormick  function #################
######## Global minimum are at (-0.54719, -1.54719), the function value is $-1.9133$.

def f35(z):
    x,y=z
    tmp=math.sin(x+y)+(x-y)*(x-y)-1.5*x+2.5*y+1
    return tmp


def f35Der(z):
    
    f35G=nd.Gradient(f35)
    tmp=f35G(z)
    return tmp

def f35Hessian(z):
    f35H=nd.Hessian(f35)
    tmp=f35H(z)
    w,v=LA.eig(tmp)
    return tmp, w

########### f36: Schaffer  function #2 #################
######## Global minimum are at (0,0), the function value is 0.

def f36(z):
    x,y=z
    tmp=0.5+(math.sin(x*x-y*y)*math.sin(x*x-y*y)-0.5)/((1+0.001*(x*x+y*y))*(1+0.001*(x*x+y*y)))
    return tmp


def f36Der(z):
    
    f36G=nd.Gradient(f36)
    tmp=f36G(z)
    return tmp

def f36Hessian(z):
    f36H=nd.Hessian(f36)
    tmp=f36H(z)
    w,v=LA.eig(tmp)
    return tmp, w


########### f37: Schaffer  function #4 #################
######## Global minimum are at (0,0), the function value is 0.

def f37(z):
    x,y=z
    tmp=0.5+(math.cos(math.sin(abs(x*x-y*y)))*math.cos(math.sin(abs(x*x-y*y)))-0.5)/((1+0.001*(x*x+y*y))*(1+0.001*(x*x+y*y)))
    return tmp


def f37Der(z):
    
    f37G=nd.Gradient(f37)
    tmp=f37G(z)
    return tmp

def f37Hessian(z):
    f37H=nd.Hessian(f37)
    tmp=f37H(z)
    w,v=LA.eig(tmp)
    return tmp, w

########### f38: Styblinski-Tang  function #################
######## Global minimum are at x_i=-2.903534.

def f38(z):
   
    tmp=np.sum(z*z*z*z-16*z*z+5*z)/2
    return tmp


def f38Der(z):
    
    f38G=nd.Gradient(f38)
    tmp=f38G(z)
    return tmp

def f38Hessian(z):
    f38H=nd.Hessian(f38)
    tmp=f38H(z)
    w,v=LA.eig(tmp)
    return tmp, w


######### f39: Regularized Log-Sum-Exp

def f39(z):
    tmp=np.log(np.sum(exp(np.matmul(cc,z))-bb))+0.5*np.sum(np.matmul(cc,z)**2)+0.5*ggam*np.sum(z**2)
    
    
    return tmp
    
def f39Der(z):
    
    f39G=nd.Gradient(f39)
    tmp=f39G(z)
    return tmp

def f39Hessian(z):
    f39H=nd.Hessian(f39)
    tmp=f39H(z)
    w,v=LA.eig(tmp)
    return tmp, w

    
########       f40: Section 8.2.7, Test Problem 7, page 112 in Hand Book of Test problems. This is a problem with simple constraints. It has 5 variables, but 3 of them can be found from the other two x3, x4. Objective function value: 0.0293. Global minimum: (x3,x4)=(1.5378, 1.9728).

def f40(z):
    x3,x4=z
    x2=x3*x3-x4+2*math.sqrt(2)-2
    x1=-x2*x2-x3*x3*x3+3*math.sqrt(2)+2
    x5=2/x1
    tmp=(x1-1)**2+(x1-x2)**2+(x2-x3)**3+(x3-x4)**4+(x4-x5)**4
    return tmp
    
def f40Der(z):
    
    f40G=nd.Gradient(f40)
    tmp=f40G(z)
    return tmp

def f40Hessian(z):
    f40H=nd.Hessian(f40)
    tmp=f40H(z)
    w,v=LA.eig(tmp)
    return tmp, w

def f40Constraint(z):
    x3,x4=z
    x2=x3*x3-x4+2*math.sqrt(2)-2
    x1=-x2*x2-x3*x3*x3+3*math.sqrt(2)+2
    x5=2/x1
    constraintx1=(x1<5) and (x1>-5)
    constraintx2=(x2<5) and (x2>-5)
    constraintx3=(x3<5) and (x3>-5)
    constraintx4=(x4<5) and (x4>-5)
    constraintx5=(x5<5) and (x5>-5)
    tmp=constraintx1 and constraintx2 and constraintx3 and constraintx4 and constraintx5
    return tmp

def f40Initialization():
    z=np.array([-5+np.random.rand()*2*5 for _ in range(2)])
    while f40Constraint(z)==False:
        z=np.array([-5+np.random.rand()*2*5 for _ in range(2)])
    tmp=z
    return tmp


########       f41: Section 2.2, Test Problem 1, page 5 in Hand Book of Test problems. This is a problem with simple constraints. It has 5 variables Objective function value: -17. Global minimum: (x1,x2,x3,x4,x5)=(1,1,0,1,0).

def f41(z):
    x1,x2,x3,x4,x5=z
    c1,c2,c3,c4,c5=42,44,45,47,47.5
    tmp=c1*x1+c2*x2+c3*x3+c4*x4+c5*x5-0.5*100*(x1**2+x2**2+x3**2+x4**2+x5**2)
    return tmp
    
def f41Der(z):
    
    f41G=nd.Gradient(f41)
    tmp=f41G(z)
    return tmp

def f41Hessian(z):
    f41H=nd.Hessian(f41)
    tmp=f41H(z)
    w,v=LA.eig(tmp)
    return tmp, w

def f41Constraint(z):
    x1,x2,x3,x4,x5=z
    constraintx1=(x1<1) and (x1>0)
    constraintx2=(x2<1) and (x2>0)
    constraintx3=(x3<1) and (x3>0)
    constraintx4=(x4<1) and (x4>0)
    constraintx5=(x5<1) and (x5>0)
    constraint1=(20*x1+12*x2+11*x3+7*x4+4*x5<40)
    tmp=constraintx1 and constraintx2 and constraintx3 and constraintx4 and constraintx5 and constraint1
    return tmp

def f41Initialization():
    z=np.array([-1+np.random.rand()*2*1 for _ in range(5)])
    while f41Constraint(z)==False:
        z=np.array([-1+np.random.rand()*2*1 for _ in range(5)])
    tmp=z
    return tmp

########       f42: Section 4.2, Test Problem 1, page 27 in Hand Book of Test problems. This is a problem with simple constraints. It has 1 variable. Objective function value: -29763.233. Global minimum: x=10. Remark: Plugging into Wolfram alpha seems to show that x=10 is not a global minimum in the given interval [-2,11]!

def f42(z):
    c6,c5,c4,c3,c2,c1,c0=1,-(52/25), (39/80), (71/10), (-79/20),(-1),(1/10)
    tmp=c6*(z**6)+c5*(z**5)+c4*(z**4)+c3*(z**3)+c2*(z**2)+c1*(z)+c0
    return tmp
    
def f42Der(z):
    
    f42G=nd.Gradient(f42)
    tmp=f42G(z)
    return tmp

def f42Hessian(z):
    f42H=nd.Gradient(f42Der)
    tmp=f42H(z)
    w=tmp
    return tmp, w

def f42Constraint(z):
    
    constraintx1=(z<11) and (z>-2)
    
    tmp=constraintx1
    return tmp

def f42Initialization():
    z=np.array([-2+np.random.rand()*13])
    while f42Constraint(z)==False:
        z=np.array([-2+np.random.rand()*13])
    tmp=z
    return tmp
    

########       f43: Section 4.3, Test Problem 2, page 27 in Hand Book of Test problems. This is a problem with simple constraints. It has 1 variable. Objective function value: -663.5. Global minimum: x=1.0911.

def f43(z):
    amatrix=np.array([-500.0,2.5,1.666666666,1.25,1.0,0.8333333, 0.714285714,
0.625,0.555555555,1.0, -43.6363636,0.41666666,0.384615384,
0.357142857,0.3333333,0.3125,0.294117647, 0.277777777, 0.263157894,
0.25,0.238095238,0.227272727,0.217391304, 0.208333333, 0.2,
0.192307692,0.185185185,0.178571428, 0.344827586, 0.6666666,
-15.48387097,0.15625,0.1515151,0.14705882,0.14285712,
0.138888888,0.135135135,0.131578947,0.128205128,0.125,
0.121951219,0.119047619,0.116279069,0.113636363,0.1111111,
0.108695652,0.106382978,0.208333333,0.408163265,0.8])
    cmatrix=np.array([amatrix[[i]]*(z**(i+1)) for i in range(50)])
    tmp=np.sum(cmatrix)
    return tmp
    
def f43Der(z):
    
    f43G=nd.Gradient(f43)
    tmp=f43G(z)
    return tmp

def f43Hessian(z):
    f43H=nd.Gradient(f43Der)
    tmp=f43H(z)
    w=tmp
    return tmp, w

def f43Constraint(z):
    
    constraintx1=(z<2) and (z>1)
    
    tmp=constraintx1
    return tmp

def f43Initialization():
    z=np.array([1+np.random.rand()*1])
    while f43Constraint(z)==False:
        z=np.array([1+np.random.rand()*1])
    tmp=z
    return tmp
    

########       f44: Section 4.6, Test Problem 5, page 29 in Hand Book of Test problems. This is a problem with simple constraints. It has 2 variables Objective function value: 0. Global minimum: (x1,x2)=(0,0).

def f44(z):
    x1,x2=z
    
    tmp=2*(x1**2)-1.05*(x1**4)+(1/6)*(x1**6)-x1*x2+x2**2
    return tmp
    
def f44Der(z):
    
    f44G=nd.Gradient(f44)
    tmp=f44G(z)
    return tmp

def f44Hessian(z):
    f44H=nd.Hessian(f44)
    tmp=f44H(z)
    w,v=LA.eig(tmp)
    return tmp, w

def f44Constraint(z):
    x1,x2=z
    constraintx1=(x1<5) and (x1>-5)
    constraintx2=(x2<5) and (x2>-5)
    
    tmp=constraintx1 and constraintx2
    return tmp

def f44Initialization():
    z=np.array([-5+np.random.rand()*2*5 for _ in range(2)])
    while f44Constraint(z)==False:
        z=np.array([-5+np.random.rand()*2*5 for _ in range(2)])
    tmp=z
    return tmp

########       f45: Section 4.10, Test Problem 9, page 31 in Hand Book of Test problems. This is a problem with simple constraints. It has 2 variables Objective function value: -5.50796. Global minimum: (x1,x2)=(2.3295,3.17846). 

def f45(z):
    x1,x2=z
    
    tmp=-x1-x2
    return tmp
    
def f45Der(z):
    
    f45G=nd.Gradient(f45)
    tmp=f45G(z)
    return tmp

def f45Hessian(z):
    f45H=nd.Hessian(f45)
    tmp=f45H(z)
    w,v=LA.eig(tmp)
    return tmp, w

def f45Constraint(z):
    x1,x2=z
    constraintx1=(x1<3) and (x1>0)
    constraintx2=(x2<4) and (x2>0)
    constraint1= (2+2*(x1**4)-8*(x1**3)+8*(x1**2)>x2)
    constraint2= (4*(x1**4)-32*(x1**3)+88*(x1**2)-96*x1+36>x2)
    tmp=constraintx1 and constraintx2 and constraint1 and constraint2
    return tmp

def f45Initialization():
    z=np.array([np.random.rand()*3, np.random.rand()*4])
    while f45Constraint(z)==False:
        z=np.array([np.random.rand()*3, np.random.rand()*4])
    tmp=z
    return tmp

########       f46: This is from the paper Toy model for protein folding, Physical Review E, August 1993.

def f46(z):
    V1=0
    for i in range(1,dimN-1):
        #print("i=", i)
        V1=V1+(1/4)*(1-math.cos(z[[i]]))

    angleMatrix=np.zeros((dimN,dimN))

    for i in range(dimN-2):
        for k in range(i+1,dimN):
            #print("i=",i)
            #print("k=",k)
            angleMatrix[i][k]=np.sum(z[i+1:k+1])

    angleMatrixCos=np.zeros((dimN,dimN))
    for i in range(dimN-2):
        for j in range(i+1,dimN):
            angleMatrixCos[i][j]=np.cos(angleMatrix[i][j])
    
    angleMatrixSin=np.zeros((dimN,dimN))
    for i in range(dimN-2):
        for j in range(i+1,dimN):
            angleMatrixSin[i][j]=np.sin(angleMatrix[i][j])
    

    rMatrix=np.zeros((dimN,dimN))
    rMatrixC=np.zeros((dimN,dimN))
    rMatrixS=np.zeros((dimN,dimN))
    for i in range(dimN-2):
        #print("i=",i)
        for j in range(i+2,dimN):
            #print("j=",j)
            #print("angleMatrixCos[i][i+1:j]=",angleMatrixCos[i][i+1:j])
            rMatrixC[i][j]=np.sum(angleMatrixCos[i][i+1:j])
            rMatrixS[i][j]=np.sum(angleMatrixSin[i][i+1:j])
            #for k in range(i+1,j):
            #    rMatrixC[i][j]=angleMatrixCos[i][k]+rMatrixC[i][j]
            #    rMatrixS[i][j]=angleMatrixSin[i][k]+rMatrixS[i][j]
            rMatrix[i][j]=((rMatrixC[i][j])**2)+((rMatrixS[i][j])**2)
                
    #print("z00=",z00_old)
    
    #print("angleMatrix=",angleMatrix)

    #print("angleMatrixCos=", angleMatrixCos)

    #print("angleMatrixSin= ",angleMatrixSin)
    
    #print("rMartix=",rMatrix)
    
    V2=0
    for i in range(dimN-2):
        for j in range(i+2,dimN):
            V2=V2+4*((rMatrix[i][j])**(-6))-(1/2)*(1+xi[[i]]+xi[[j]]+5*xi[[i]]*xi[[j]])*((rMatrix[i][j])**(-3))
    tmp=V1+V2
    
    #print("angleMatrix=",angleMatrix)

    #print("angleMatrixCos=", angleMatrixCos)

    #print("angleMatrixSin= ",angleMatrixSin)
    
    #print("rMartix=",rMatrix)

    return tmp
    
def f46Der(z):
    
    f46G=nd.Gradient(f46)
    tmp=f46G(z)
    return tmp

def f46Hessian(z):
    f46H=nd.Hessian(f46)
    tmp=f46H(z)
    w,v=LA.eig(tmp)
    return tmp, w

def f46Constraint(z):
    tmp=True
    for i in range(dimN):
        tmp=tmp and (z[i]>0) and (z[i]<2*math.pi)
    return tmp

def f46Initialization():
    z=np.array([np.random.rand()*2*math.pi for _ in range(dimN)])
    xi=np.array([np.random.randint(2) for _ in range(dimN)])
    for i in range(dimN):
        if xi[[i]]==0:
            xi[[i]]=-1
    tmpZ=z
    tmpXi=xi
    return tmpZ, tmpXi


###############

########### f47: Griewank problem #################
########


def f47(z):
    normz=np.sum(z**2)
    #print("normz=",normz)
    countN=np.array([z[i]/math.sqrt(i+1) for i in range(dimN)])
    productCos=1
    for i in range(dimN):
        productCos=productCos*math.cos(countN[i])
    tmp=1+(1/4000)*normz-productCos
    tmp=tmp
    #print("check what xi", xi)
    return tmp


def f47Der(z):
    f47G=nd.Gradient(f47)
    tmp=f47G(z)
    #tmp1=tmp[0:dimN]
    return tmp

def f47Hessian(z):
    f47H=nd.Hessian(f47)
    tmp=f47H(z)
    #tmp1=tmp[0:dimN,0:dimN]
    w,v=LA.eig(tmp)
    return tmp, w


################

#z00=np.array([-32+random()*64 for _ in range(D)])
#z00=np.array([23.49261912, -13.86471849])

#print(z00)
#print(f23(z00))
#print(f23Der(z00))
#print(f23Hessian(z00))

#print(f23(z00).dtype)
#print(type(f23(z00)))


#print(f23Der(z00).dtype)
#print(type(f23Der(z00)))






#Newton's method:
def NewtonMethod():
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
    




######## Note: For the function f39 (taken from the paper by Rodomanov-Nesterov), cc is a mmxN matrix, entries randomly chosen in [-1,1], bb is a mm matrix, entries randomly chosen in [-1,1], the initial point z00_old is randomly chosen with length 1/N. The parameters for the function are bb, cc and ggam.




D=10
mm=1

dimN=D


bound=5

ggam=1






#verbose=True
#stopCriterion=1

printLastPoint=True

printHessianEigenvalues=True

verbose=False
#stopCriterion=0
stopCriterion=0
#stopCriterion=2

NIterate=10000
#NIterate=10000

print("atol=", atol)
aatol=1e-20

#z00_rand=np.array([-1+np.random.rand()*2*1 for _ in range(dimN)])
#z00_old=z00_rand/(math.sqrt(L2Norm2(z00_rand,dimN)*dimN))
#z00_old=z00_rand

#z00_old=np.array([np.random.uniform(-1,1)*math.pi for _ in range(dimN)])

z00_old=np.array([10 for _ in range(dimN)])
#z00_old=np.array([20*np.random.rand()-10 for _ in range(dimN)])


bb=np.array([-1+np.random.rand()*2*1 for _ in range(mm)])

cc1=np.random.rand(mm,dimN)
cc2=cc1*2
cc=cc2-1

rrtol=1e-10


print("the function is", "f47")
f=f47
fDer=f47Der
fHessian=f47Hessian



    
#z00_old, xi=f46Initialization()
#xi=np.array([1,1,1,-1,1])
#print("Matrix xi=",xi)
#tmp=f46(z00_old)
#print("f46 value=",tmp)


def constraintChect(z):
    tmp=f46Constraint(z)
    return tmp

#z00InPaper=np.array([0,0.29723*math.pi,0.33306*math.pi,0.62176*math.pi,0])
#z00InPaper=np.array([0,0.61866*math.pi,0])
#print("Optimum according to the paper=", f46(z00InPaper))

#x1=np.random.rand()*2*math.pi
#x2=0
#x3=np.random.rand()*2*math.pi
#z00_old=np.array([x1,x2,x3])


#print("angleMatrix=",angleMatrix)

#print("angleMatrixCos=", angleMatrixCos)

#print("angleMatrixSin= ",angleMatrixSin)

#z00_old=np.array([-2.903534+0.3, -2.903534-0.8])

# For the function f40, bb takes values in {-1,1}, and bb and cc and z00_old are taken from real datasets: #https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/

#z00_old=np.array([-2.903534+0.3, -2.903534-0.8])

#z00_old=np.array([0.26010457, -10.91803423, 2.98112261, -15.95313456,  -2.78250859, -0.77467653,  -2.02113182,   9.10887908, -10.45035903,  11.94967756, -1.24926898,  -2.13950642,   7.20804014,   1.0291962,    0.06391697, 2.71562242, -11.41484204,  10.59539405,  12.95776531,  11.13258434,
 #  8.16230421, -17.21206152,  -4.0493811,  -19.69634293,  14.25263482, 3.19319406,  11.45059677,  18.89542157,  19.44495031,  -3.66913821])

#z00_old=np.array([-0.15359941, -0.59005902, 0.45366905, -0.94873933,  0.52152264, -0.02738085,0.17599868,  0.36736119,  0.30861332,  0.90622707,  0.10472251, -0.74494753, 0.67337336, -0.21703503, -0.17819413, -0.14024491, -0.93297061,  0.63585997, -0.34774991, -0.02915787, -0.17318147, -0.04669807,  0.03478713, -0.21959983,
 # 0.54296245,  0.71978214, -0.50010954, -0.69673303,  0.583932,   -0.38138978, -0.85625076,   0.20134663, -0.71309977, -0.61278167,  0.86638939,  0.45731164, -0.32956812,  0.64553452, -0.89968231,  0.79641384,  0.44785232,  0.38489415, -0.51330669,  0.81273771, -0.54611157, -0.87101225, -0.72997209, -0.16185048, 0.38042508, -0.63330049,  0.71930612, -0.33714448, -0.24835364, -0.78859559,
 #-0.07531072,  0.19087508, -0.95964552, -0.72759281,  0.13079216,  0.6982817, 0.54827214,  0.70860856, -0.51314115, -0.54742142,  0.73180924, -0.28666226, 0.89588517,  0.35797497, -0.21406766, -0.05558283,  0.89932563, -0.16479757, -0.29753867,  0.5090385,   0.95156811,  0.8701501,   0.62499125, -0.22215331, 0.8355082,  -0.83695582, -0.96214862, -0.22495384, -0.30823426,  0.55635375,
 # 0.38262606, -0.60688932, -0.04303575,  0.59260985,  0.5887739,  -0.00570958, -0.502354, 0.50740011, -0.08916369,  0.62672251,  0.13993309, -0.92816931, 0.50047918,  0.856543,    0.99560466, -0.44254687])

print("Number of iterates=", NIterate)

print("initial point=", z00_old)
#print("The matrix xi=",xi)

## For some problems, such as f39 and f40, we also use the minimum point, for another stopping criterion
zmin=np.zeros(D)
errtol=rrtol*abs(f(z00_old)-f(zmin))
print("errtol=",errtol)

#printHessianEigenvalues=True
tmp, w=fHessian(z00_old)
print("derivative at the initial point=", fDer(z00_old))
print("Eigenvalues of the Hessian at the initial point=", w)
print("function value at initial point=", f(z00_old))
#print(type(z00_old))
#print(len(z00_old))

#NewtonMethod()
#RandomNewQNewton()

#RandomNewtonMethod()
#NewQNewton()
#BFGS()
#InertialNewtonM()
#UnboundedTwoWayBacktrackingGD()
#BacktrackingGD()
#AdaptiveCubicRegularisation()
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

#zshift=np.full(D,0,dtype=np.float128)
#zsquare=np.full(D,0,dtype=np.float128)
#for i in range(D-1):
#    zshift[i]=i+1
#for i in range(D-1):
#    zsquare[i]=i*i
#print(zshift)
#print(zsquare)

#f23G=nd.Gradient(f24)
#f23H=nd.Hessian(f24)
#print(f23G(z00))
#print(f23H(z00))

#z00=np.array([1,1,1])
#print(z00)
#print(f26(z00))
#print(f26Der(z00))
#print(f26Hessian(z00))
