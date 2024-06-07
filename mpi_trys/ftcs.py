import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from matplotlib import cm
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

def create_halo(local_mat, h=0):
    # print(local_mat)
    if len(np.shape(local_mat)) == 2:
        shape = np.shape(local_mat[:,0])
        # print(f"shape : {shape}")
        local_mat = np.c_[np.zeros(shape), local_mat, np.zeros(shape)]
    elif len(np.shape(local_mat)) == 1:
        local_mat = np.r_[local_mat[0]-h, local_mat, local_mat[-1] + h  ]
    
    # print(local_mat)
    return local_mat
"""   
def exchange_vals1(local_mat,i):
    # print(f"sending from {rank}")
    # if rank==2 :print(f"#A#\nr{rank}_i{i:0=2}::\n{local_mat}\n#B#")
    np.savetxt(f"exchange/r{rank}_i{i:0=4}A.csv",local_mat)
    if rank < size - 1:
        comm.send(local_mat[:,-1], dest = rank + 1, tag = 1)
        recv_left = comm.recv(source = rank + 1, tag = 1)
        local_mat[:,-1] = recv_left
        
    
    if rank > 0:
        comm.send(local_mat[:, 0], dest = rank - 1, tag = 1)
        recv_right = comm.recv(source = rank -1, tag = 1)
        local_mat[:,0] = recv_right
        
    # if rank==2 : print(f"#A#\nr{rank}_i{i:0=2}::\n{local_mat}\n#B#")
    np.savetxt(f"exchange/r{rank}_i{i:0=4}B.csv",local_mat)
    return local_mat
"""     
def exchange_vals(local_mat,i):
    shape = np.shape(local_mat[:,0])
    sendleft = np.ascontiguousarray(local_mat[:,1])
    # print(f"{sendleft}#r{rank}_i{i}")
    sendright = np.ascontiguousarray(local_mat[:,-2])
    recvleft = np.zeros(shape)
    recvright = np.zeros(shape)
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
        local_mat[:,0] = recvleft
    if rank < size - 1:
        local_mat[:,-1] = recvright
    
    # np.savetxt(f"exchange/r{rank}_i{i:0=4}B.csv",local_mat)

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
    
    # V = create_halo(V)
    x = create_halo(x,h)
    V = np.zeros([N, np.size(x)]);
    # print(f"rank{rank}::\nx:{x}")
    
    # print(f"at rank : {rank}, len(x) : {len(x)}")
    
    
    
    # print(np.shape(V))
    # f = open(name, "w")
    # print(f'{name} file opened')
    # f2 = open("U_mat.txt", "w")
    # write initial condition in solution matrix U
    for l in range(np.size(x)):
        V[0,l] = v0(x[l])
        # f2.write(f"\n{x[l]} : {V[0,l]}")
    
    i = 0
    for lt in range(1,N):
        # plt.matshow(V)
        # plt.savefig(f"frames0of4/r{rank}_V{i:0=3}")
        # plt.clf()
        # np.savetxt(f"frames/r{rank}_V{i:0=3}.csv", V, delimiter=",")
        # i = i + 1
        comm.Barrier()
        V = exchange_vals(V,lt)
        # plt.matshow(V)
        # plt.savefig(f"frames0of4/r{rank}_V{i:0=3}")
        # plt.clf()
        # np.savetxt(f"frames/r{rank}_V{i:0=3}.csv", V, delimiter=",")
        i = i + 1
        for lx in range(1,np.size(x)-1):
            Dp = D(x[lx]+h/2) * (np.abs(x[lx]+h/2)<L)
            Dm = D(x[lx]-h/2) * (np.abs(x[lx]-h/2)<L)
        
            # f.write(f"\n({lt},{lx}):: Dp: {D(x[lx]+h/2)} * {int((np.abs(x[lx]+h/2)<L))}, Dm: {D(x[lx]-h/2)} * {int((np.abs(x[lx]-h/2)<L))}")
            # f.write(f"\n({lt},{lx}):: Dp: {Dp}, Dm: {Dm}")

            V[lt,lx] = V[lt-1,lx] + (tau/h**2)*Dp*((V[lt-1,lx+1]) - V[lt-1,lx]) + \
                                    (tau/h**2)*Dm*((V[lt-1,lx-1]) - V[lt-1,lx]) + \
                                     tau*S(x[lx], -(lt-1)*tau)
        
            
   
    
    # print(f"#rank{rank}#\nV:{V}")
    # f.close()
    # print(f'{name} file closed')
    t = np.arange(0,N)*tau;
    # keep only -L <= x <= L
    V = V[:,1:-1];
    x = x[1:-1];

    # np.savetxt(f"V_{rank}of{size}.csv", V, delimiter=",")
    return V,x,t
