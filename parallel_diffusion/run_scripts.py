import os 
import shutil
import glob
import time

nodes = [1,2]
procs = [2,4,8,16,32]
x_divs = [2,4,8,16,32,64,128]
repeat = [1,2,3]

base_path = os.path.join(os.getcwd(),"base")
# case_path = os.path.join(os.getcwd(),"results")

base_diffusion_path = os.path.join(base_path,"diffusion.py")
base_ftcs_path = os.path.join(base_path,"ftcs.py")
base_submit_script_path = os.path.join(base_path,"submit_script.sh")


for r in repeat:
    case_path = os.path.join(os.getcwd(),f"results_{r}")
    i = 0
    for p in procs:
        i = i + 1
        for d in x_divs[i-1:]:
            # make path to each case
            dir_name = os.path.join(case_path,f"p-{p}",f"divs-{d}")
            if not(os.path.isdir(dir_name)): os.makedirs(dir_name)
            # copy files from base to case
            shutil.copy(base_diffusion_path, dir_name)
            shutil.copy(base_ftcs_path, dir_name)
            shutil.copy(base_submit_script_path, dir_name)
            # shutil.copy(os.path.join(base_path,"sleep.py"), dir_name)
            node = nodes[1] # 2
            if p < 8:
                node = nodes[0] # 1
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
            # os.chdir(dir_name)
            # os.system("sbatch submit_script.sh")
            # count = 0
            # while not(glob.glob("*.png") and count < 20): # wait 10 seconds max
            #     time.sleep(0.5)
            #     count = count + 1
            
            # print(f"continuing {p}-{d}")






        
        
            

        

