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
D=pr.D

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
    for i in range(D-1):
        zshift[i]=z[i+1]
    z1=z-onesvector
    tmp=np.sum(100*((zshift-z)**2)+z1**2)-100*(z[D-1]**2)-z1[D-1]**2
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
def NewtonMethod(f,fDer, z00_old, NIterate,fHessian,dimN,verbose ):
    print("0.Newton's method:")

    z00=z00_old
    print(z00)

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
            print(z00)

    print(z00)
    print(f(z00))
    print(fDer(z00))
    return

#Random Newton's method:

def RandomNewtonMethod(f, fDer, fHessian,z00_old,NIterate,  dimN,verbose):

    print("1.Random Newton's method:")

    z00=z00_old
    print(z00)

    minLR = 0
    maxLR = 2


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
            print(z00)

    print(z00)
    print(f(z00))
    print(fDer(z00))
    return
    
    
#NewQNewton

def NewQNewton(f,fDer,fHessian,z00_old ,NIterate,dimN,verbose):

    print("2. New Q Newton's method:")

    z00=z00_old
    delta=1

    print(z00)
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
            print(z00)

    print(z00)
    print(f(z00))
    print(fDer(z00))
    return
    
    
    
#Random New Q Newton's method

def RandomNewQNewton(f, fDer, fHessian, z00_old, NIterate, dimN, verbose):

    print("3. Random New Q Newton's method:")
    z00=z00_old
    print(z00)

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
            print(z00)

    print(z00)
    print(f(z00))
    print(fDer(z00))
    return
#BFGS
def BFGS(f, fDer, NIterate, z00_old):

    print("4. BFGS")
    z00=z00_old
    print(z00)

    xopt=fmin_bfgs(f, z00,fprime=fDer,maxiter=NIterate,gtol=pr.rtol)
    print(xopt)

    return

#Backtraking Gradient Descent

def BacktrackingGD(f, fDer, z00_old, fHessian, NIterate, dimN, verbose):

    print("5. Backtracking Gradient Descent")

    z00=z00_old
    print(z00)
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
            print(z00)
    
    print(z00)
    print(f(z00))
    print(fDer(z00))
    return
#Inertial Newton's method

def InertialNewtonM(f, fDer, fHessian, z00_old, NIterate, dimN, verbose):
    print("6. Inertial Newton's method")

    z00=z00_old

    print(z00)

    alpha1 = 0.5

    beta1 = 0.1

    lambda1 = (1/beta1) -alpha1



    psin=np.array([-1+random()*2 for _ in range(dimN)])






    for m in range(NIterate):

        gn = fDer(z00)

        gamman=(m+1)**(-0.5)

    

        z00 = z00 + gamman * (lambda1*z00-(1/beta1)*psin-beta1*gn)

        psin=psin+gamman*(lambda1*z00-(1/beta1)*psin)
        if verbose:
            print(z00)
    print(z00)
    print(f(z00))
    print(fDer(z00))
    return
    

def main():
    D=2
    dimN=D
    bound=5
    verbose=False
    NIterate=10
    #z00_old=np.array([-bound+random()*2*bound for _ in range(dimN)])
    z00_old=np.array([1.02183524, 0.13979978])

    print("Number of iterates=", NIterate)

    print("initial point=", z00_old)

    print("the function is", "f38")
    f=f38
    # calcutale derivation of the 38th function
    fDer=f38Der
    # calculate Hessinan matrix
    fHessian=f38Hessian
    # Newton method
    print("*"*30,"the Newton method","*"*30)
    NewtonMethod(f,fDer, z00_old, NIterate,fHessian,dimN,verbose )
    # RandomNewton method
    print("*"*30,"the RandomNewton method","*"*30)
    RandomNewtonMethod(f, fDer, fHessian,z00_old,NIterate,  dimN,verbose)
    # New Q-Newton method
    print("*"*30,"the New Q-Newton method","*"*30)
    NewQNewton(f,fDer,fHessian,z00_old ,NIterate,dimN,verbose)
    # Random New Q-Newton method
    print("*"*30,"the Random New Q-Newton method","*"*30)
    RandomNewQNewton(f, fDer, fHessian, z00_old, NIterate, dimN, verbose)
    # Back tracking GD method
    print("*"*30,"the Back tracking GD method","*"*30)
    BacktrackingGD(f, fDer, z00_old, fHessian, NIterate, dimN, verbose)
    # BFGS method
    print("*"*30,"the BFGS method","*"*30)
    BFGS(f, fDer, NIterate, z00_old)
    # inertial Newton method
    print("*"*30,"the inertial Newton method","*"*30)
    InertialNewtonM(f, fDer, fHessian, z00_old, NIterate, dimN, verbose)

if __name__ == '__main__':
  main()


#f23G=nd.Gradient(f24)
#f23H=nd.Hessian(f24)
#print(f23G(z00))
#print(f23H(z00))

#z00=np.array([1,1,1])
#print(z00)
#print(f26(z00))
#print(f26Der(z00))
#print(f26Hessian(z00))

