import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from matplotlib import cm
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

def create_halo(local_mat, h):
    if len(np.shape(local_mat)) == 2:
        shape = np.shape(local_mat[:,0])
        # print(f"shape : {shape}")
        local_mat = np.c_[np.zeros(shape), local_mat, np.zeros(shape)]
    elif len(np.shape(local_mat)) == 1:
        local_mat = np.r_[local_mat[0]-h, local_mat, local_mat[-1] + h  ]
    
    # print(f"rank : {rank}")
    # print(f"local_mat(x) : {local_mat}")
    return local_mat
    
def exchange_vals(local_mat):
    # print(f"sending from {rank}")
    if rank < size - 1:
        comm.send(local_mat[:,-1], dest = rank + 1, tag = 1)
        recv_left = comm.recv(source = rank + 1, tag = 1)
        local_mat[:,-1] = recv_left
    
    if rank > 0:
        comm.send(local_mat[:, 0], dest = rank - 1, tag = 1)
        recv_right = comm.recv(source = rank -1, tag = 1)
        local_mat[:,0] = recv_right
        
    return local_mat

def ftcs(L,N,tau,D,S,v0,name="untitled.txt"):
    # x = np.linspace(-L-h,L+h, np.ceil(2*L/h).astype('int')+2)
    global_array_size = 128 # <-- 128 (or 16) should be any power of 2?
    x = np.zeros(global_array_size//size)
    h = 0
    if rank == 0:
        # h = np.gradient(np.linspace(-L,L,global_array_size))[1] 
        h = (2*L/global_array_size)
        print(f"h : {h}")
        global_x = np.linspace(-L,L, np.ceil(2*L/h).astype('int'))
        print(f"len(x) : {len(x)}")
        global_t = np.zeros(1)
        global_V = np.zeros(1)
    else:
        global_x = np.zeros(1)
    
    h = comm.bcast(h,root=0)
    # print(f"at rank {rank}, h : {h}")
    comm.Scatter(global_x, x, root = 0)
    x = create_halo(x,h)
    # print(f"at rank : {rank}, len(x) : {len(x)}")
    V = np.zeros([N, np.size(x)]);
    
    
    # print(np.shape(V))
    # f = open(name, "w")
    # print(f'{name} file opened')
    # f2 = open("U_mat.txt", "w")
    # write initial condition in solution matrix U
    for l in range(np.size(x)):
        V[0,l] = v0(x[l])
        # f2.write(f"\n{x[l]} : {V[0,l]}")
    
    # print(f"rank : {rank} | {V[0,:]}")
    
    # print("f2 written V")
    # V = exchange_vals(V)
    # print(V)
    # f2.close()
    # iterate ftcs method
    i = 0
    for lt in range(1,N):
        for lx in range(1,np.size(x)-1):
            Dp = D(x[lx]+h/2) * (np.abs(x[lx]+h/2)<L)
            Dm = D(x[lx]-h/2) * (np.abs(x[lx]-h/2)<L)
        
            # f.write(f"\n({lt},{lx}):: Dp: {D(x[lx]+h/2)} * {int((np.abs(x[lx]+h/2)<L))}, Dm: {D(x[lx]-h/2)} * {int((np.abs(x[lx]-h/2)<L))}")
            # f.write(f"\n({lt},{lx}):: Dp: {Dp}, Dm: {Dm}")

            V[lt,lx] = V[lt-1,lx] + (tau/h**2)*Dp*((V[lt-1,lx+1]) - V[lt-1,lx]) + \
                                    (tau/h**2)*Dm*((V[lt-1,lx-1]) - V[lt-1,lx]) + \
                                     tau*S(x[lx], -(lt-1)*tau)
            comm.Barrier()
            V = exchange_vals(V)
        
    # f.close()
    # print(f'{name} file closed')
    t = np.arange(0,N)*tau;
    # keep only -L <= x <= L
    V = V[:,1:-1];
    x = x[1:-1];

    return V,x,t
