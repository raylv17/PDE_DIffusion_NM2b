import os 
import shutil
import glob
import time

# DO NOT CHANGE
nodes  = [1,2,3]
x_divs = [8,16,32,64,128,192]

#USER INPUT
procs  = [2,4,8,16,32,64]
repeat = [1]

base_path = os.path.join(os.getcwd(),"base")
# case_path = os.path.join(os.getcwd(),"results")

base_diffusion_path = os.path.join(base_path,"diffusion.py")
base_ftcs_path = os.path.join(base_path,"ftcs.py")
base_submit_script_path = os.path.join(base_path,"submit_script.sh")


for r in repeat:
    case_path = os.path.join(os.getcwd(),f"results_{r}")
    print(f"case_number: {case_path[-1]}")
    i = 0
    for p in procs:
        if p >= 16:
            # to keep current divs > current number of procs used.
            i = i + 1 
            sliced_divs = x_divs[i:] 
        else:
            sliced_divs = x_divs[:]
        
        for d in sliced_divs:
            # make path to each case
            dir_name = os.path.join(case_path,f"p-{p}",f"divs-{d}")
            if not(os.path.isdir(dir_name)): os.makedirs(dir_name)
            # copy files from base to case
            shutil.copy(base_diffusion_path, dir_name)
            shutil.copy(base_ftcs_path, dir_name)
            shutil.copy(base_submit_script_path, dir_name)
            # shutil.copy(os.path.join(base_path,"sleep.py"), dir_name)
            node = nodes[0] # 1 [2,4]
            if  8 <= p <= 32:
                node = nodes[1] # 2 [8, 16, 32]
            elif p > 32:
                node = nodes[2] # 3 [64]
            # modification of params on each case
            dir_name = os.path.join(case_path,f"p-{p}",f"divs-{d}")
            case_submit_script_path = os.path.join(dir_name, "submit_script.sh")
            with open(case_submit_script_path, "r") as file:
                filedata = file.read()

            filedata = filedata.replace("_procs", str(p)).replace("_node", str(node))

            with open(case_submit_script_path, "w") as file:
                file.write(filedata)

            dir_name = os.path.join(case_path,f"p-{p}",f"divs-{d}")
            case_diffusion_path = os.path.join(dir_name, "diffusion.py")
            with open(case_diffusion_path, "r") as file:
                filedata = file.read()

            filedata = filedata.replace("_divs", str(d))

            with open(case_diffusion_path, "w") as file:
                file.write(filedata)

            file.close()

            # run case inside each case directory
            os.chdir(dir_name)
            # if case already executed previously, don't resubmit
            if not os.path.isfile(os.path.join(os.getcwd(),f"p-{p}.jpg")):
                os.system("sbatch submit_script.sh")
            else:
                print(f"skipping: {r}-procs{p}-divs{d}, it's already finished")
            
            # wait for case to be finished before executing another 
            # (checks whether .png was generated)
            count = 0
            while not(glob.glob("*.png")): # wait 20 seconds max (give time-out error)
                time.sleep(1)
                count = count + 1
                if count > 20: # works for tau = 0.001
                    print("time-out")
                    break
            
            print(f"    {r}-procs{p}-divs{d}-node{node} submitted")

        os.chdir(case_path)
        os.chdir("..")

print("creating strong_scaling plots")
os.system("python plots.py")

# Organize results
# compiled_results_folder_name = "results"
# compiled_plots_folder_name = "plots"
# if not os.path.isdir(compiled_results_folder_name):
#     os.mkdir("results")
# if not os.path.isdir(compiled_plots_folder_name):
#     os.mkdir("plots")

# results_dirs = glob.glob("results_*")
# plots_dirs = glob.glob("plot_*")

# for res_dir in results_dirs:
#     shutil.move(res_dir,compiled_results_folder_name)

# for plt_dir in plots_dirs:
#     shutil.move(plt_dir,compiled_plots_folder_name)


