import os
import shutil
import matplotlib.pyplot as plt

procs = [2,4,8,16,32]
xdivs = [2,4,8,16,32,64,128]
results = [1,2,3]

path = os.path.join(os.getcwd())
results_dir = "results_compiled"
if not(os.path.isdir(results_dir)):
    os.mkdir(results_dir)


# collect out file from each case
for r in results:
    i = 0
    for p in procs:
        for d in xdivs[i:]:
            case_name = os.path.join(f"results_{r}",f"p-{p}", f"divs-{d}")
            case_dir = os.path.join(path,case_name)
            new_out = f"out-{r}-{p}-{d}.txt"
            shutil.copy(os.path.join(case_dir,"out.txt"), os.path.join(case_dir, new_out))
            # shutil.copy(os.path.join(case_dir, new_out), os.path.join(case_dir, "out.txt"))
            # os.remove(os.path.join(case_dir,new_out))
            if not(os.path.isfile(os.path.join(results_dir, new_out))):
                shutil.move(os.path.join(case_dir, new_out), results_dir)
            
        i = i + 1