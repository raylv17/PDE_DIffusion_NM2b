import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from matplotlib import cm

def ftcs(L,N,h,tau,D,S,v0,name="untitled.txt"):
    x = np.linspace(-L-h,L+h, np.ceil(2*L/h).astype('int')+2)
    # print(x)
    V = np.zeros([N, np.size(x)]);
    # print(np.shape(V))
    f = open(name, "w")
    print(f'{name} file opened')
    f2 = open("U_mat.txt", "w")
    # write initial condition in solution matrix U
    for l in range(np.size(x)):
        V[0,l] = v0(x[l])
        f2.write(f"\n{x[l]} : {V[0,l]}")
        
    print("f2 written V")
    
    f2.close()
    # iterate ftcs method
    for lt in range(1,N):
        for lx in range(1,np.size(x)-1):
            Dp = D(x[lx]+h/2) * (np.abs(x[lx]+h/2)<L)
            Dm = D(x[lx]-h/2) * (np.abs(x[lx]-h/2)<L)
        
            # f.write(f"\n({lt},{lx}):: Dp: {D(x[lx]+h/2)} * {int((np.abs(x[lx]+h/2)<L))}, Dm: {D(x[lx]-h/2)} * {int((np.abs(x[lx]-h/2)<L))}")
            f.write(f"\n({lt},{lx}):: Dp: {Dp}, Dm: {Dm}")

            V[lt,lx] = V[lt-1,lx] + (tau/h**2)*Dp*((V[lt-1,lx+1]) - V[lt-1,lx]) + \
                                    (tau/h**2)*Dm*((V[lt-1,lx-1]) - V[lt-1,lx]) + \
                                     tau*S(x[lx], -(lt-1)*tau)
#             print(lx)
    f.close()
    print(f'{name} file closed')
    t = np.arange(0,N)*tau;
    # keep only -L <= x <= L
    V = V[:,1:-1];
    x = x[1:-1];

    return V,x,t
