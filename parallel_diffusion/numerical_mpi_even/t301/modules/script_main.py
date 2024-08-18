from modules.ext_imports import *
from input_params import *

base_path = os.path.join(os.getcwd(),"base")
base_diffusion_path = os.path.join(base_path,"diffusion.py")
base_ftcs_path = os.path.join(base_path,"ftcs.py")
base_submit_script_path = os.path.join(base_path,"submit_script.sh")

def copy_files_base_to_case(dir_name):
    shutil.copy(base_diffusion_path, dir_name)
    shutil.copy(base_ftcs_path, dir_name)
    shutil.copy(base_submit_script_path, dir_name)

def assign_node(p):
    node = 1 # 1 [2,4]
    if  8 <= p <= 32:
        node = 2 # 2 [8, 16, 32]
    elif p > 32:
        node = 3 # 3 [64]
    return node

def modify_case_params(case_path,dir_name,node,p,d):
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

    filedata = filedata.replace("_divs", str(d)).replace("_tdivs", str(int(2/tau)))

    with open(case_diffusion_path, "w") as file:
        file.write(filedata)

    file.close()

def run_case(dir_name,r,p,d,node):
    os.chdir(dir_name)
    # if case already executed previously, don't resubmit
    if not os.path.isfile(os.path.join(os.getcwd(),f"p-{p}.jpg")):
        os.system("sbatch submit_script.sh")
        print(f"    {r}-procs{p}-divs{d}-node{node} submitted")
    else:
        print(f"skipping: {r}-procs{p}-divs{d}, it's already finished")
        return
    
    # wait for case to be finished before executing another 
    # (i.e. checks whether .png was generated)
    count = 0
    if tau == 0.0001: 
        timer = 120
    else: # tau = 0.001
        timer = 20
    
    # print(os.getcwd())
    while not(glob.glob("*.png")): # wait 20 seconds max (give time-out error)
        time.sleep(1)
        count = count + 1
        if count > timer: # works for tau = 0.001
            print("time-out, submitting next case")
            return
