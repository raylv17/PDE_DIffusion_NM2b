from modules.script_plot import *
from input_params import *
procs = np.array(procs)

# Collect results from each case to a result_dir (initialized in plots_script.py)
collect_outfiles()
out_files = create_out_file_list()

# Create plots directory
times_per_div_dict, max_time = create_times_per_div_dict(out_files)
make_plots_dir(plots_dir)

# Generate Times for each coarseness
for d in x_divs:
    gen_execution_time(d,times_per_div_dict)
plt.close()
print(f"time plots for each coarseness at {plots_dir}")

# Generate Strong Scaling Plots
mean_list = []
for d in x_divs:
    mean = gen_scaling_plot(d,times_per_div_dict)
    proc_list = procs[np.where(procs<=d)]
    mean_list.append((procs[np.where(procs<=d)],mean))
print(f"scaling plots created at {plots_dir}/")

# Generate plot for constant proc and increasing div size
times_per_proc_dict = create_times_per_proc_dict(out_files)
gen_step_size_time_plot(out_files, times_per_proc_dict, max_time)
print(f"time/step_size plots created at {plots_dir}/")

# collect results in folder
# organize_session_results(session_name)
