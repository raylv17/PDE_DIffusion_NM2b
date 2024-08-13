# 2d distribution
# global array (1d with initial conditios)
# local arrays (2d empty array)
# local arrays will receive chunks of the elements of global array at the first row
# local_array[0,:] = chunk of global for particular rank.

from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

u0 = lambda x : np.where(abs(x) < 1.5,1,0)
diffusion  = lambda x : 1   # Diffusion coefficient function
source  = lambda x,t : 0 # source function

def distribute_and_initialize(h: float, tau: float, length: float, time: float):
    """
    descretizes space and time
    distributes descritization to other procs (all procs have halo)
    each proc then implements the initial boundary condition
    """

    if rank == 0:    
        print(f"params: h:{h}, tau:{tau}, length:{length}, time:{time}")
        ## descretization
        global_space_size = int(2*length/h)
        global_time_size = int(time/tau)
        # global_space_array = np.linspace(-length,length,global_space_size, dtype='f') # descretized space
        global_space_array = np.arange(-length,length,h) # descretized space
        # print(global_space_array)
        # global_recv_mat = np.empty(global_time_size*global_space_size,dtype='f')
        global_mat = np.zeros([global_time_size, global_space_size]) # solution matrix
        local_array_size = int(global_space_size/size)
        first_slice_index = local_array_size + global_space_size%size # proc0 keeps remainder
        print(f"px_local_mat_size: {local_array_size}, p0_local_mat_size: {first_slice_index}")
        
        # allocate space at proc0
        local_space_array = np.zeros(first_slice_index)
        local_mat = np.zeros([global_time_size, first_slice_index+2]) # each proc has a halo
        # intialize values for local_space_array at proc0
        local_space_array = global_space_array[:first_slice_index]
        local_mat[0,1:-1] = u0(local_space_array) # [x u(0) x]

        # distribution to other procs
        for r in range(1,size):
            comm.Send([global_space_array[first_slice_index+(local_array_size*(r-1)):first_slice_index+(local_array_size*r)],MPI.FLOAT], r)
    else:
        # h = None
        # tau = None
        local_array_size = None
        local_space_array = None
        local_mat = None
        global_time_size = None
        # global_recv_mat = None
        global_mat = None
    
    # h = comm.bcast(h, root=0)
    # tau = comm.bcast(tau, root=0)
    local_array_size = comm.bcast(local_array_size, root=0)
    global_time_size = comm.bcast(global_time_size, root=0)

    if rank != 0:
        # allocation
        local_mat = np.zeros([global_time_size, local_array_size+2])
        local_space_array = np.zeros(local_array_size)
        # recieve from proc0 and intialization based on boundary condition
        comm.Recv([local_space_array, MPI.FLOAT], source=0)
        local_mat[0,1:-1] = u0(local_space_array)
    
    # print(f"{rank} local array size: {np.shape(local_mat)} " )
    return local_mat, local_space_array, global_mat

def share_border_vals(row_index: int,local_mat: np.ndarray) -> np.ndarray:
    # buffers
    send_rbuf = np.empty(1)
    recv_rbuf = np.empty(1)
    send_lbuf = np.empty(1)
    recv_lbuf = np.empty(1)
    # print(previous_row,local_mat[previous_row,-2])
    if rank < size-1: # all right border exchanges
        # needs to be array (even if one element) for sendrecvs to work
        send_rbuf[0] = local_mat[row_index,-2] 
        comm.Send([send_rbuf,MPI.FLOAT],dest=rank+1,tag=0)
        comm.Recv([recv_rbuf,MPI.FLOAT],source=rank+1,tag=1)
        local_mat[row_index,-1] = recv_rbuf[0]
    if rank > 0: # all left border exchanges
        send_lbuf[0] = local_mat[row_index,1]
        comm.Send([send_lbuf,MPI.FLOAT],dest=rank-1,tag=1)
        comm.Recv([recv_lbuf,MPI.FLOAT],source=rank-1,tag=0)
        local_mat[row_index,0] = recv_lbuf[0]
    return local_mat

def ftcs(local_mat, local_x, length, time, h, tau):
#     for lt = 2:N
#       for lx = 2:length(x)-1
#          Dp = D(x(lx)+h/2) * ( abs(x(lx)+h/2)<L );
# 	     Dm = D(x(lx)-h/2) * ( abs(x(lx)-h/2)<L );

# 	     U(lt,lx) = U(lt-1,lx) + tau/h^2*Dp * (U(lt-1,lx+1) - U(lt-1,lx)) ...
# 		                       + tau/h^2*Dm * (U(lt-1,lx-1) - U(lt-1,lx)) ...
# 			                   + tau* S(x(lx),(lt-1)*tau);
#       end
#    end
    # print(f"{rank} :: {len(local_mat[0,:])-1}")
    for lt in range(local_mat.shape[0]):
        # print("hello")
        local_mat = share_border_vals(lt-1, local_mat)
        for lx in range(1,len(local_mat[0,1:-1])-1):
            Dp = diffusion(local_x[lx] + h/2) * (abs(local_x[lx] + h/2 < length))
            Dm = diffusion(local_x[lx] - h/2) * (abs(local_x[lx] - h/2 < length))

            local_mat[lt,lx] = local_mat[lt-1,lx] + tau/h**2*Dp * ( local_mat[lt-1,lx+1] - local_mat[lt-1,lx] ) \
                                                  + tau/h**2*Dm * ( local_mat[lt-1,lx-1] - local_mat[lt-1,lx] ) \
                                                  + tau *  source(local_x[lx+1],(lt-1)*tau )
            # print(rank, lt,lx, local_mat[lt,lx])

    return local_mat[:,1:-1]

def gather(local_mat,global_mat):
    """
    argument takes local_mat without halo
    """
    send_mat = local_mat.flatten()
    # print("types:",rank,local_mat.dtype, send_mat.dtype)
    global_recv_mat = None
    # print(f"{rank} | {np.shape(local_mat)} local_mat")
    # print(f"{rank} | {np.shape(send_mat)} send_mat")
    
    if rank == 0:
        # print(f"proc 0, global_mat_flatten: | {np.shape(global_mat.flatten())}")
        global_recv_mat = np.zeros(len(send_mat)*size)
        # print(global_mat.shape)

    # Due to uneven local_mat shape at proc0, hard to Gather.
    # solution may be Gatherv!
    comm.Gather(send_mat, global_recv_mat, root=0)

    if rank == 0:
        # print(f"global_flat: {np.shape(global_recv_mat)} {global_recv_mat}")
        # print(global_recv_mat, global_recv_mat.shape)

        for i in range(size):
            global_mat[:,local_mat.shape[1]*i:local_mat.shape[1]+local_mat.shape[1]*i] = global_recv_mat[i*len(send_mat):(i+1)*len(send_mat)].reshape(local_mat.shape)
        
        return global_mat
        

    # if rank==0:
    # global_mat = np.empty([global_rows_size,global_cols_size], dtype='i')
    # print(global_recv_mat, np.shape(global_recv_mat))
    # col = -1
    # for i in range(5*size):
    #     row = i%5
    #     if row == 0:
    #         col = col + 1
    #     # print(row,5*col,5*(col+1),global_recv_mat[5*i:5*(i+1)])
    #     global_mat[row,5*col:5*(col+1)] = global_recv_mat[5*i:5*(i+1)]

    # print(global_recv_mat, np.shape(global_recv_mat))

