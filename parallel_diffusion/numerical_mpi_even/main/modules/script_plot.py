from modules.ext_imports import *
from input_params import *

session_name = "session1"
path = os.path.join(os.getcwd())
results_dir = f"{delta_t}_outs_collected"
plots_dir = f"{delta_t}"
x_divs = np.array(x_divs)

def make_results_dir():
    if not(os.path.isdir(results_dir)):
        os.mkdir(results_dir)


def collect_outfiles():
    make_results_dir()
    for r in range(1,repeat+1):
        i = 0
        for p in procs:
            sliced_divs = x_divs[np.where(x_divs>=p)]
            for d in sliced_divs:
                case_name = os.path.join(f"results_{r}",f"p-{p}", f"divs-{d}")
                case_dir = os.path.join(path,case_name)
                new_out = f"out-{r}-{p}-{d}.txt"
                # print(case_dir)
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
    
def create_out_file_list():
    outlist = glob.glob(f"{results_dir}/out-*")
    out_files = []
    for outfile in outlist:
        r_val = int(outfile[:-4].split("-")[1:][0]) # repeats
        p_val = int(outfile[:-4].split("-")[1:][1]) # procs
        d_val = int(outfile[:-4].split("-")[1:][2]) # divs (# of discretized cells)
        out_files.append((r_val, p_val, d_val))
    out_files.sort()
    # print(out_case_names)
    return out_files

# prepare times dictionary for each case.
def create_times_per_div_dict(out_files : list):
    times_per_div = {}
    for r,p,d in out_files:
        case=f"{r}-{d}"
        times_per_div[case] = []
    
    max_time = 0
    for r,p,d in out_files:
        case=f"{r}-{d}"
        read_file = os.path.join(results_dir,f"out-{r}-{p}-{d}.txt")
        if (os.path.isfile(os.path.join(read_file))):
            with open(read_file) as file:
                read = file.read()
                index = read.find("time: ")
                case_time = float(read[index:].split()[1])
                if case_time > max_time:
                    max_time = case_time
                times_per_div[case].append(case_time)
    
    # print(max_time)
    return times_per_div, max_time

def gen_execution_time(div : int, times_per_div : dict):
    fig, ax = plt.subplots()
    ax.grid(True)
    ax.set_title(f"Execution Time | Divisions of L :{div} ~ h :{10/div:.3f}")
    ax.set_xlabel("# of procs")
    ax.set_ylabel("time [s]")
    plt.xticks(rotation=60, fontsize=8)
    x = np.array([[]])
    set_axis = 1
    for case in times_per_div.keys():
        if int(case.split("-")[1]) == div:
            # print(case, times[case])
            x = np.concatenate( (x,np.array( [times_per_div[case]])) , axis=set_axis)
            set_axis=0
            ax.set_xticks([0]+[i for i in range(2,2**len(times_per_div[case])+2,2)])
            procs = [2**i for i in range(0,len(times_per_div[case]))]
            plt.plot(procs,np.array(times_per_div[case]), "-o", label=f"{case}")
    for i,j in zip(procs,x.mean(axis=0)):
        ax.annotate(f"{j:1.2f}",xy=(i+0.1,j))
    plt.plot(procs,x.mean(axis=0), "k--o", label="mean")
    # print(div,x)

    plt.legend()
    plt.savefig(os.path.join(plots_dir, f"d{div}-plot"), dpi=300)


def make_plots_dir(plots_dir):
    if not(os.path.isdir(plots_dir)):
        os.mkdir(plots_dir)

def create_times_per_proc_dict(out_files : list):
    times_per_proc = {}
    
    for r,p,d in out_files:
        case=f"{r}-{p}"
        times_per_proc[case] = []
    
    for r,p,d in out_files:
        case=f"{r}-{p}"
        read_file = os.path.join(results_dir,f"out-{r}-{p}-{d}.txt")
        if (os.path.isfile(os.path.join(read_file))):
            with open(read_file) as file:
                read = file.read()
                index = read.find("time: ")
                times_per_proc[case].append(float(read[index:].split()[1]))

    return times_per_proc


def gen_step_size_time_plot(out_files : list, times_per_proc_dict : dict, max_time: float):
    if tau == 0.0001:
        sec_step_ticks = 64
        ax_step_ticks = 32
        axy_ticks = 50
        axy_round = -2
    else: # tau == 0.001
        sec_step_ticks = 16
        ax_step_ticks = 16
        axy_ticks = 2
        axy_round = -1
    # collect x_divs from out_files
    x_divs = np.array([]).astype(int)
    for r,p,d in out_files:
        if r == out_files[0][0] and p == out_files[0][1]:
            x_divs = np.append(x_divs, d)
    
    max_time = max(times_per_proc_dict['1-1'])
    # generate plot
    plt.clf()
    fig, ax = plt.subplots()
    # ax.set_yticks(range(0,int(round(max_time, axy_round)),axy_ticks))
    ax.set_xticks(range(0,x_divs[-1]+1,ax_step_ticks))
    ax.set_xticklabels(labels=range(0,x_divs[-1]+1,ax_step_ticks), rotation=60, fontsize=8)
    sec = ax.secondary_xaxis(location=1)
    sec.set_xticks(range(0,x_divs[-1]+1,sec_step_ticks))
    sec.set_xticklabels(labels= ["inf"] + [f"\n{i:.3f}" for i in 10/np.arange(16,x_divs[-1]+1,sec_step_ticks)], 
                        rotation=10,
                        fontsize=8)
    for case_name,times in times_per_proc_dict.items():
        repeat = case_name.split("-")[0]
        if repeat == "1":
            proc = int(case_name.split("-")[1])
            # increase starting value of x_div based on proc
            case_divs = x_divs[np.where(x_divs >= proc)]
            ax.plot(case_divs, np.array(times),"-o",markersize=3, label=f"procs-{proc}")

    ax.set_xlabel("# decretized cells")
    ax.set_ylabel("time [s]")
    sec.set_xlabel("step-size [h]")

    # plt.text(203, max_time, f'{max_time:.2f}s')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(delta_t,"const_proc_plot"), dpi=300)

def gen_scaling_plot(div : int, times_per_div : dict):
    fig, ax = plt.subplots()
    ax.grid(True)
    ax.set_xscale("log", base=2)
    ax.set_title(f"Strong Scaling | Divisions of L :{div} ~ h :{10/div:.3f}")
    ax.set_xlabel("# of procs")
    ax.set_ylabel("speed up [t(1)/t(n)]")
    plt.xticks(rotation=0, fontsize=8)
    x = np.array([[]])
    set_axis = 1
    for case in times_per_div.keys():
        if int(case.split("-")[1]) == div:
            # print(case, times[case])
            x = np.concatenate( (x,np.array( [times_per_div[case]])) , axis=set_axis)
            set_axis=0
            # ax.set_xticks([0]+[i for i in range(2,2**len(times_per_div[case])+2,2)])
            procs = [2**i for i in range(0,len(times_per_div[case]))]
            # label_text = 
            plt.plot(procs,np.array(times_per_div[case][0])/np.array(times_per_div[case]), "-o", label=f"session-{int(case.split('-')[0])}")
    for i,j in zip(procs,x.mean(axis=0)[0]/x.mean(axis=0)):
        ax.annotate(f"{j:1.2f}",xy=(i+0.1,j))
    plt.plot(procs,x.mean(axis=0)[0]/x.mean(axis=0), "k--o", label="mean")
    # print(div,x)

    plt.legend()
    plt.savefig(os.path.join(plots_dir, f"scaling{div}-plot"), dpi=300)

    return x.mean(axis=0)/x.mean(axis=0)[0]


def organize_session_results(session_name):
    files = glob.glob("plot*") + glob.glob("results*") + glob.glob("__pychache__")
    if len(files) == 0:
        print("no files to collect, please rerun simulation")
        return
    
    if not os.path.isdir(session_name):
        os.mkdir(session_name)
    
    for file in files:
        shutil.move(file, session_name)


