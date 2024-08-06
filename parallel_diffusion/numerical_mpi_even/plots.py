import os
import shutil
import matplotlib.pyplot as plt
import glob
import numpy as np
from input_params import *

# x_divs = [128]
# procs = [4]
# repeat = [1]


path = os.path.join(os.getcwd())
results_dir = f"{delta_t}_results_compiled"
if not(os.path.isdir(results_dir)):
    os.makedirs(results_dir)

x_divs = np.array(x_divs)
# collect out file from each case
for r in range(1,repeat+1):
    i = 0
    for p in procs:
        sliced_divs = x_divs[np.where(x_divs>=p)]
        for d in sliced_divs:
            case_name = os.path.join(f"results_{r}",f"p-{p}", f"divs-{d}")
            case_dir = os.path.join(path,case_name)
            new_out = f"out-{r}-{p}-{d}.txt"
            if (os.path.isfile(os.path.join(case_dir,"out.txt"))):
                shutil.copy(os.path.join(case_dir,"out.txt"), results_dir)
            else:
                print(f"{new_out} does not exist")
            # shutil.copy(os.path.join(case_dir, new_out), os.path.join(case_dir, "out.txt"))
            if (os.path.isfile(os.path.join(results_dir, "out.txt"))):
                shutil.move(os.path.join(results_dir, "out.txt"), os.path.join(results_dir,new_out))
            else:
                print(f"{new_out} skipped")
            
        i = i + 1

# prepare times dictionary for each case.
times = {}
for r in range(1,repeat+1):
    i = 0
    for p in procs:
        for d in x_divs[i:]:
            case= f"{r}-{d}"
            times[case] = []
        i = i + 1

# collect times for each case
for r in range(1,repeat+1):
    i = 0
    for p in procs:
        for d in x_divs[i:]:
            out_file = os.path.join(results_dir,f"out-{r}-{p}-{d}.txt")
            if (os.path.isfile(os.path.join(out_file))):
                with open(out_file) as file:
                    case= f"{r}-{d}"
                    read = file.read()
                    index = read.find("time: ")
                    times[case].append(float(read[index:].split()[1]))
        i = i + 1

plots_dir = f"{delta_t}"
if not(os.path.isdir(plots_dir)):
    os.mkdir(plots_dir)

def print_plots(div):
    fig, ax = plt.subplots()
    ax.grid(True)
    ax.set_title(f"Strong Scaling plot | Divisions of L :{div}")
    ax.set_xlabel("# of procs")
    ax.set_ylabel("simulation time [s]")
    plt.xticks(rotation=60, fontsize=8)
    x = np.array([[]])
    set_axis = 1
    for case in times.keys():
        if int(case.split("-")[1]) == div:
            # print(case, times[case])
            x = np.concatenate( (x,np.array( [times[case]])) , axis=set_axis)
            set_axis=0
            ax.set_xticks([i for i in range(2,2**len(times[case])+2,2)])
            procs = [2**i for i in range(1,len(times[case])+1)]
            plt.plot(procs,times[case], "-o", label=f"{case}")
    plt.plot(procs,x.mean(axis=0), "k--o", label="mean")
    # print(div,x)

    plt.legend()
    plt.savefig(os.path.join(plots_dir, f"{div}-plot"), dpi=300)


for d in x_divs:
    print_plots(d)

print(f"plots created at {results_dir}")
