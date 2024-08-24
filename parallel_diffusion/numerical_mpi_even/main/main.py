from modules.script_main  import *

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
            copy_files_base_to_case(dir_name)
            
            # assign appropriate node
            node = assign_node(p)
            
            # modification of params on each case
            modify_case_params(case_path,dir_name,node,p,d)

            # run case inside each case directory
            run_case(dir_name,r,p,d,node)

        # os.chdir(case_path)
        # os.chdir("..")

time.sleep(3)
user_input = input("create plots? (y or n)") 
if user_input.upper() == "Y":
    print("creating plots")
    os.system("python run_plots.py")
else:
    print("skipping plots, to generate later please run:\n    python run_plots.py")

