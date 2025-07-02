import os
import subprocess

experiment_series = "comparing_to_exact_solution"
scripts_dir = "scripts/{}".format(experiment_series)
working_dir = "/nfs/home/aandreev/evo_network"

script_files = ['exact_sat_watts_strogatz_uniform_1-2_uniform_0_75-1_40_8_0_5_123.py', 'exact_sat_barabasi_albert_uniform_1-5_uniform_0_75-1_50_4.py', 'exact_sat_watts_strogatz_uniform_1-2_const_0_8_40_8_0_5_123.py', 'exact_sat_barabasi_albert_uniform_1-5_const_0_8_50_4.py', 'tdg_watts_strogatz_uniform_1-2_uniform_0_75-1_40_8_0_5_123.py', 'tdg_barabasi_albert_uniform_1-5_uniform_0_75-1_50_4.py', 'tdg_watts_strogatz_uniform_1-2_const_0_8_40_8_0_5_123.py', 'tdg_barabasi_albert_uniform_1-5_const_0_8_50_4.py', '1+1_10000__watts_strogatz_uniform_1-2_uniform_0_75-1_40_8_0_5_123.py', '1+1_10000__barabasi_albert_uniform_1-5_uniform_0_75-1_50_4.py', '1+1_10000__watts_strogatz_uniform_1-2_const_0_8_40_8_0_5_123.py', '1+1_10000__barabasi_albert_uniform_1-5_const_0_8_50_4.py', 'tdg_1+1_10000__watts_strogatz_uniform_1-2_uniform_0_75-1_40_8_0_5_123.py', 'tdg_1+1_10000__barabasi_albert_uniform_1-5_uniform_0_75-1_50_4.py', 'tdg_1+1_10000__watts_strogatz_uniform_1-2_const_0_8_40_8_0_5_123.py', 'tdg_1+1_10000__barabasi_albert_uniform_1-5_const_0_8_50_4.py']

for script_name in script_files:
    script_path = os.path.join(scripts_dir, script_name)
    log_path = script_path.replace('.py', '.log')

    print(f"Running {script_name}...")

    with open(log_path, 'w') as log_file:
        process = subprocess.Popen(
            ['python', script_path],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=working_dir
        )
        retcode = process.wait()

    if retcode == 0:
        print(f"[DONE] {script_name} completed successfully.")
    else:
        print(f"[ERROR] {script_name} exited with code {retcode}. See {log_path} for details.")

print("All experiments completed.")
