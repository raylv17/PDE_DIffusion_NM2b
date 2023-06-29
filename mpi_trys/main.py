
from dist_mpi import *

# sayHello()

N = 32
L = 8
h = 0.2


D = lambda x : 1
S = lambda x,t : 0

# eachproc(N)
# Determine_Value(L,h,D)


local_mat = make_mat(5)


local_mat = create_halo(local_mat)
# print(local_mat)
# print()

# for i in range(2):  
    # print(i)
    # swap_mat = exchange_vals(local_mat)
    # print(swap_mat)
    # print()

swap_mat = exchange_vals(local_mat)
print(f"matrix at {rank}")
print(swap_mat)
gather_mat = comm.gather(local_mat[:,1:-1], root = 0)

if rank == 0:
    print(f"{rank}")
    # print(gather_mat)
    print(np.shape(gather_mat))
    gather_mat = np.concatenate(gather_mat, axis=1)
    print(np.shape(gather_mat))
    print(gather_mat)