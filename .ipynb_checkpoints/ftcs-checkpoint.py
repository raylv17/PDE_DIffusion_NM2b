import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from matplotlib import cm

def ftcs(L,N,h,tau,D,S,v0):
    x = np.linspace(-L-h,L+h, np.ceil(2*L/h).astype('int')+2)
    V = np.zeros([N, np.size(x)]);
#     print(np.shape(V))

    # write initial condition in solution matrix U
    for l in range(np.size(x)):
        V[0,l] = v0(x[l])

    # iterate ftcs method
    for lt in range(N-1):
        for lx in range(np.size(x)-1):
            Dp = D(x[lx]+h/2) * (np.abs(x[lx]+h/2)<L)
            Dm = D(x[lx]-h/2) * (np.abs(x[lx]-h/2)<L)

            V[lt+1,lx] = V[lt,lx] + (tau/h**2)*Dp*((V[lt,lx+1]) - V[lt,lx]) + \
                                    (tau/h**2)*Dm*((V[lt,lx-1]) - V[lt,lx]) + \
                                     tau*S(x[lx], -(lt)*tau)
#             print(lx)
    t = np.arange(0,N)*tau;
    # keep only -L <= x <= L
    V = V[:,1:-1];
    x = x[1:-1];

    return V,x,t
