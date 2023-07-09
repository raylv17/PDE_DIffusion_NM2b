
from dist_mpi import *

# sayHello()

N = 32
L = 8
h = 0.2


D = lambda x : 1
S = lambda x,t : 0

# eachproc(N)
# Determine_Value(L,h,D)


local_mat = make_mat(3)


local_mat = create_halo(local_mat)
# print(f"original_matrix at {rank}")
# print(local_mat)
# print(local_mat)
# print()

# for i in range(2):  
    # print(i)
    # swap_mat = exchange_vals(local_mat)
    # print(swap_mat)
    # print()

for i in range(1):
    # local_mat = manipulate_mat(local_mat)
    local_mat = exchange_vals(local_mat)
    print(f"changed{i}_matrix at {rank}")
    print(local_mat)



gather_mat1 = comm.gather(local_mat, root = 0)
gather_mat2 = comm.gather(local_mat[:,1:-1], root = 0)

if rank == 0:
    print(f"{rank}")
    # print(gather_mat1)
    # print(np.shape(gather_mat1))
    gather_mat1 = np.concatenate(gather_mat1, axis=1)
    print(np.shape(gather_mat1))
    print(gather_mat1)
    
    gather_mat2 = np.concatenate(gather_mat2, axis=1)
    print(np.shape(gather_mat2))
    print(gather_mat2)