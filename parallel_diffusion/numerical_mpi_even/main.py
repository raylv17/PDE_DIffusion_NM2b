import os 
import shutil
import glob
import time
import numpy as np
from input_params import *

# USER INPUT
# x_divs = [128]
# procs  = [4]
# repeat = [1]

####
base_path = os.path.join(os.getcwd(),"base")
base_diffusion_path = os.path.join(base_path,"diffusion.py")
base_ftcs_path = os.path.join(base_path,"ftcs.py")
base_submit_script_path = os.path.join(base_path,"submit_script.sh")

x_divs = np.array(x_divs)
for r in range(1,repeat+1):
    case_path = os.path.join(os.getcwd(),f"results_{r}")
    print(f"case_number: {case_path[-1]}")
    for p in procs:
        sliced_divs = x_divs[np.where(x_divs>=p)]
        
        for d in sliced_divs:
            # make path to each case
            dir_name = os.path.join(case_path,f"p-{p}",f"divs-{d}")
            if not(os.path.isdir(dir_name)): os.makedirs(dir_name)
            # copy files from base to each case
            shutil.copy(base_diffusion_path, dir_name)
            shutil.copy(base_ftcs_path, dir_name)
            shutil.copy(base_submit_script_path, dir_name)
            
            # assign appropriate node
            node = 1 # 1 [2,4]
            if  8 <= p <= 32:
                node = 2 # 2 [8, 16, 32]
            elif p > 32:
                node = 3 # 3 [64]
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
                print(f"    {r}-procs{p}-divs{d}-node{node} submitted")
            else:
                print(f"skipping: {r}-procs{p}-divs{d}, it's already finished")
                continue
            
            # wait for case to be finished before executing another 
            # (i.e. checks whether .png was generated)
            count = 0
            # print(os.getcwd())
            while not(glob.glob("*.png")): # wait 20 seconds max (give time-out error)
                time.sleep(1)
                count = count + 1
                if count > 20: # works for tau = 0.001
                    print("time-out, submitting next case")
                    break
        
        os.chdir(case_path)
        os.chdir("..")

time.sleep(3)
user_input = input("create plots? (y or n)") 
if user_input.upper() == "Y":
    print("creating plots")
    os.system("python run_plots.py")
else:
    print("skipping plots, to generate later please run:\n    python run_plots.py")


