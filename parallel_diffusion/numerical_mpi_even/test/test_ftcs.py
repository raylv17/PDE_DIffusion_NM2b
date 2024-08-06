import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

def create_halo(local_mat, h):
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
    # shape = np.shape(local_mat[:,0])[0] # equal to global_t
    # sendleft = np.ascontiguousarray(local_mat[lt-1:lt,1])
    sendleft  = local_mat[lt-1:lt,1] # exchange single val at lt-1 as an array
    sendright = local_mat[lt-1:lt,-2]
    # print(f"rank {rank}: send_size: {lt}")
    recvleft = np.zeros(1) # allocate receivers
    recvright = np.zeros(1)
    # print(local_mat)
    if rank > 0:
        # print(recvleft)
        comm.Sendrecv(sendleft, rank - 1, 0, recvleft, rank - 1)
        # print(recvleft)
    if rank < size - 1:
        comm.Sendrecv(sendright, rank + 1, 0,recvright, rank + 1)
        # print(recvright, sendright)
        # print(recvright)

    # np.savetxt(f"exchange/r{rank}_i{i:0=4}A.csv",local_mat)
    # print(rank)
    if rank > 0:
        local_mat[lt-1:lt,0] = recvleft
    if rank < size - 1:
        local_mat[lt-1:lt,-1] = recvright
    
    # np.savetxt(f"exchange/r{rank}_i{i:0=4}B.csv",local_mat)

    return local_mat

def ftcs(L,N,global_array_size,tau,D,S,v0,name="untitled.txt"):
    """
    global array size: must be divisible by total number of ranks i.e. {size}
    x = np.linspace(-L-h,L+h, np.ceil(2*L/h).astype('int')+2)
    
    """
    ## allocate space of local_x in all ranks
    x = np.zeros(global_array_size//size) 
    if rank == 0:
        # h = np.gradient(np.linspace(-L,L,global_array_size))[1] 
        h = (2*L/global_array_size) # later broadcasted
        # print(f"step_size [h] : {h}")
        global_x = np.linspace(-L,L, np.ceil(2*L/h).astype('int'))
        print(f"size of local_x : {len(x)}")
        # global_t = np.zeros(1)
        # global_V = np.zeros(1)
    else:
        global_x = np.zeros(1)
        h = 0
    
    h = comm.bcast(h,root=0)
    comm.Scatter(global_x, x, root = 0)
    x = create_halo(x,h) # [x-h x x+h] # values at borders to implement boundary condition (u(x))
    V = np.zeros([N, np.size(x)]) # part of solution matrix at rank with halo
    
    
    # print(np.shape(V))
    # f = open(name, "w")
    # print(f'{name} file opened')
    # f2 = open("U_mat.txt", "w")
    # write initial condition in solution matrix U
    for l in range(np.size(x)):
        V[0,l] = v0(x[l])
        # f2.write(f"\n{x[l]} : {V[0,l]}")
    

    for lt in range(1,N):
        V = exchange_vals(V,lt)
        for lx in range(1,np.size(x)-1):
            Dp = D(x[lx]+h/2) * (np.abs(x[lx]+h/2)<L)
            Dm = D(x[lx]-h/2) * (np.abs(x[lx]-h/2)<L)
        
            # f.write(f"\n({lt},{lx}):: Dp: {D(x[lx]+h/2)} * {int((np.abs(x[lx]+h/2)<L))}, Dm: {D(x[lx]-h/2)} * {int((np.abs(x[lx]-h/2)<L))}")
            # f.write(f"\n({lt},{lx}):: Dp: {Dp}, Dm: {Dm}")

            V[lt,lx] = V[lt-1,lx] + (tau/h**2)*Dp*((V[lt-1,lx+1]) - V[lt-1,lx]) + \
                                    (tau/h**2)*Dm*((V[lt-1,lx-1]) - V[lt-1,lx]) + \
                                     tau*S(x[lx], -(lt-1)*tau)
        comm.Barrier()
    
   
    # np.savetxt(f"{rank}_V.csv", V, delimiter=",")

    # f.close()
    # print(f'{name} file closed')
    t = np.arange(0,N)*tau;
    # keep only -L <= x <= L
    V = V[:,1:-1];
    x = x[1:-1];

    return V,x,t
