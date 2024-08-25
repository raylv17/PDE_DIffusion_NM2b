## USER INPUT for tau = 0.001
tau = 0.001
x_divs = [8, 16, 32, 64, 128, 192]
procs  = [1,2,4,8,16,32,64]
repeat = 3
delta_t = "plot_t301"

# USER INPUT for tau = 0.0001
# tau = 0.0001
# x_divs = [8, 16, 32, 64, 128, 256, 512, 576, 640, 704]
# procs  = [1,2,4,8,16,32,64]
# repeat = 3
# delta_t = "plot_t401"

## SAMPLE: SAMPLE SMALL INPUT
# tau = 0.001
# x_divs = [128]
# procs  = [8]
# repeat = 1
# delta_t = "plot_p8_d128"


if __name__ == "__main__":
    import numpy as np
    MAX_PROC_SIZE = 64
    tau = 0.0001 # or 0.001
    
    # to make an array for x_divs
    powers = np.array([i for i in range(3,40)])
    divs = np.array(2**powers)

    # all powers of 2 that fulfill statiblity conditon base on given tau
    index = np.where(tau/(10/divs)**2 < 0.5)
    divs = divs[index]
    print(divs)

    # all multiples of MAX_PROC_SIZE that fulfill statiblity condition based on given tau
    # these are all the multiples within 2**(highest_power) and 2**(highest_power+1)
    highest_power = powers[index][-1]
    # print(highest_power)
    multiples = [i for i in range(2**highest_power, 2**(highest_power+1), MAX_PROC_SIZE) if tau/(10/i)**2 < 0.5]
    print(multiples)

    # # combine to form x_divs
    x_divs = list(divs)[:-1] + list(multiples)
    print(x_divs)
    
    # shows the corresponding step sizes based on length L=10
    print(10/np.array(x_divs))

