from script_plot import *
from input_params import *
# Collect results from each case to a result_dir (initialized in plots_script.py)

collect_outfiles()
out_files = create_out_file_list()

# Generate Strong Scaling Plots
times_per_div_dict = create_times_per_div_dict(out_files)
make_plots_dir(plots_dir)
for d in x_divs:
    gen_scaling_plots(d,times_per_div_dict)

print(f"scaling plots created at {plots_dir}/")

# Generate plot for constant proc and increasing div size
times_per_proc_dict = create_times_per_proc_dict(out_files)
gen_step_size_time_plot(out_files, times_per_proc_dict)
print(f"time/step_size plots created at {plots_dir}/")

# collect results in folder
# organize_session_results(session_name)
