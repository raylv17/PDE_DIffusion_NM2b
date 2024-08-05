from diffusion import *
import matplotlib.pyplot as plt

h = 0.1 # step_size (space)
tau = 0.001  # step_size (time)
LENGTH = 5  # distance from origin (meters) # actual domain length = 2*LENGTH
TIME = 2  # time to simulate (seconds)

[local_mat,local_x, global_mat] = distribute_and_initialize(h,tau,LENGTH,TIME)

local_mat = ftcs(local_mat, local_x, LENGTH, TIME, h, tau)
# print(rank, np.shape(local_mat), local_mat)

gather(local_mat, global_mat)


if rank == 0:
    print(global_mat.shape)
    plt.plot(np.arange(-LENGTH,LENGTH, h), global_mat[-1,:])
    plt.show()