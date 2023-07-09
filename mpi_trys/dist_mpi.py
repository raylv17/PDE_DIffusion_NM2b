# distributing data to n processors

from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size 

def sayHello():
    print(f"Hello from {rank} out of {size}")
    
    
def Determine_Value(L,h,D):
    x = np.linspace(-L-h,L+h,np.ceil(2*L/h).astype('int')+2)
    local_n = int(np.size(x)/size)
    for lx in range(rank*local_n+1,(rank+1)*local_n+1):
        Dp = D(x[lx]+h/2) * (np.abs(x[lx]+h/2)<L)
        Dm = D(x[lx]-h/2) * (np.abs(x[lx]-h/2)<L)
        print(f"({rank},{lx}):: Dp: {Dp}, Dm: {Dm}")
    print()    
    
def eachproc(N):
    local_n = int(N/size)
    local_array = np.zeros(local_n).astype(int)
    for i in range(rank*local_n, (rank+1)*local_n):
        # print(i)
        # local_array.append(i)
        # print(i % size)
        local_array[i % local_n] = i
        
    print(f"rank {rank} : {local_array}")
    
def make_mat(shape):
    # shape = 6
    local_mat = np.ones((shape,shape)).astype(int) * rank
    return local_mat
    
def create_halo(local_mat):
    shape = np.shape(local_mat[:,0])
    extra_col = np.ones(shape).astype(int)*rank
    local_mat = np.c_[extra_col, local_mat, extra_col]
    
    # print(local_mat)
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
    
    
def manipulate_mat(local_mat):
    shape = np.shape(local_mat)
    print(shape)
    for i in range(shape[0]):
        for j in range(1, shape[1]-1):
            local_mat[i,j] = local_mat[i,j] + 1
            
    return local_mat