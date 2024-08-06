import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from matplotlib import cm
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

v0 = lambda x : float(abs(x)<1.5)
D = lambda x : 1
S = lambda x,t : 0

def create_halo(local_mat, h):
    """
    allocates space at borders of each local_mat for exchanging values with neighbouring rank
    here, only elif is executed:
        - the argument local_mat is 1D, and the values at the borders are set to 
            - local_mat[0]-h ad local_[-1]+h respectively
        - these values overlap the values at the borders of neighbouring ranks
            - for e.g. rank_left = [h1, rank_left, h2] and [h2, rank_mid, h3] and [h3, rank_right, h4]
        - these values used to:
            - perform correct boundary conditions before ftcs is implemented
            - perform exchanges between values at borders between neighbouring ranks during ftcs
    """
    # works even for 2D local_mats, but in our case, only elif will run
    if len(np.shape(local_mat)) == 2:
        shape = np.shape(local_mat[:,0])
        # print(f"shape : {shape}")
        local_mat = np.c_[np.zeros(shape), local_mat, np.zeros(shape)]
    elif len(np.shape(local_mat)) == 1: # for row matrices e.g [a,b,c,..] np.shape returns (1,) instead of (1,0)
        local_mat = np.r_[local_mat[0]-h, local_mat, local_mat[-1] + h  ]
    
    # print(f"rank : {rank}")
    # print(f"local_mat(x) : {local_mat}")
    return local_mat
    
def exchange_vals(local_mat: np.ndarray,lt: int):
    """
    exchange single values at lt-1 as an array
    send and recv values neighbouring(left/right) rank
    """
    # prepare value to send
    sendleft  = local_mat[lt-1:lt,1] 
    sendright = local_mat[lt-1:lt,-2]
    # allocate receivers
    recvleft = np.zeros(1) 
    recvright = np.zeros(1)

    # send and receive from left rank
    if rank > 0:
        comm.Sendrecv(sendleft, rank - 1, 0, recvleft, rank - 1)
    # send and receive from right rank
    if rank < size - 1:
        comm.Sendrecv(sendright, rank + 1, 0,recvright, rank + 1)

    # insert received values back to local solution matrix of current rank
    if rank > 0:
        local_mat[lt-1:lt,0] = recvleft
    if rank < size - 1:
        local_mat[lt-1:lt,-1] = recvright

    return local_mat

def ftcs(L,N,global_array_size,tau,name="untitled.txt"):
    """
    global array size: must be divisible by total number of ranks i.e. {size}
    x = np.linspace(-L-h,L+h, np.ceil(2*L/h).astype('int')+2)
    
    """
    ## allocate space of local_x in all ranks
    x = np.zeros(global_array_size//size) 

    if rank == 0:
        h = (2*L/global_array_size)
        global_x = np.linspace(-L,L, np.ceil(2*L/h).astype('int'))
        print(f"size of local_x : {len(x)}")
    else:
        global_x = np.zeros(1)
        h = 0
    
    h = comm.bcast(h,root=0)
    comm.Scatter(global_x, x, root = 0) # distribution of global_x to each local_x
    x = create_halo(x,h) # [x-h x x+h] values at borders to implement boundary condition (v0(x))
    V = np.zeros([N, np.size(x)]) # part of solution matrix at rank with halo

    # boundary condition
    for l in range(np.size(x)):
        V[0,l] = v0(x[l])

    # ftcs method
    for lt in range(1,N):
        V = exchange_vals(V,lt)
        for lx in range(1,np.size(x)-1):
            Dp = D(x[lx]+h/2) * (np.abs(x[lx]+h/2)<L)
            Dm = D(x[lx]-h/2) * (np.abs(x[lx]-h/2)<L)

            V[lt,lx] = V[lt-1,lx] + (tau/h**2)*Dp*((V[lt-1,lx+1]) - V[lt-1,lx]) + \
                                    (tau/h**2)*Dm*((V[lt-1,lx-1]) - V[lt-1,lx]) + \
                                     tau*S(x[lx], -(lt-1)*tau)
        comm.Barrier()
    
   
    # np.savetxt(f"{rank}_V.csv", V, delimiter=",")

    # f.close()
    # print(f'{name} file closed')
    t = np.arange(0,N)*tau
    # keep only -L <= x <= L
    V = V[:,1:-1]
    x = x[1:-1]

    return V,x,t
