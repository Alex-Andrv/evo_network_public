import os
import subprocess

scripts_dir = "scripts/sensitivity_analysis"
working_dir = "/Users/aandreev/PycharmProjects/evo_network_public"

script_files = ['tdg_iter_descent_v4_10000, 1, 25__ca-GrQc_uniform_1-1000_const_0_8.py', 'tdg_iter_descent_v4_10000, 1, 25__ca-GrQc_uniform_1-1000_uniform_0_75-1.py', 'tdg_iter_descent_v4_10000, 1, 25__facebook_combined_uniform_1-1000_const_0_8.py', 'tdg_iter_descent_v4_10000, 1, 25__facebook_combined_uniform_1-1000_uniform_0_75-1.py', 'tdg_iter_descent_v4_10000, 1, 25__p2p-Gnutella06_uniform_1-1000_const_0_8.py', 'tdg_iter_descent_v4_10000, 1, 25__p2p-Gnutella06_uniform_1-1000_uniform_0_75-1.py', 'tdg_iter_descent_v4_10000, 1, 25__p2p-Gnutella08_uniform_1-1000_uniform_0_75-1.py', 'tdg_iter_descent_v4_10000, 1, 25__p2p-Gnutella08_uniform_1-1000_const_0_8.py', 'tdg_iter_descent_v4_10000, 1, 25__Wiki-Vote_uniform_1-1000_const_0_8.py', 'tdg_iter_descent_v4_10000, 1, 25__Wiki-Vote_uniform_1-1000_uniform_0_75-1.py', 'tdg_iter_descent_v4_10000, 1, 50__ca-GrQc_uniform_1-1000_const_0_8.py', 'tdg_iter_descent_v4_10000, 1, 50__ca-GrQc_uniform_1-1000_uniform_0_75-1.py', 'tdg_iter_descent_v4_10000, 1, 50__facebook_combined_uniform_1-1000_const_0_8.py', 'tdg_iter_descent_v4_10000, 1, 50__facebook_combined_uniform_1-1000_uniform_0_75-1.py', 'tdg_iter_descent_v4_10000, 1, 50__p2p-Gnutella06_uniform_1-1000_const_0_8.py', 'tdg_iter_descent_v4_10000, 1, 50__p2p-Gnutella06_uniform_1-1000_uniform_0_75-1.py', 'tdg_iter_descent_v4_10000, 1, 50__p2p-Gnutella08_uniform_1-1000_uniform_0_75-1.py', 'tdg_iter_descent_v4_10000, 1, 50__p2p-Gnutella08_uniform_1-1000_const_0_8.py', 'tdg_iter_descent_v4_10000, 1, 50__Wiki-Vote_uniform_1-1000_const_0_8.py', 'tdg_iter_descent_v4_10000, 1, 50__Wiki-Vote_uniform_1-1000_uniform_0_75-1.py', 'tdg_iter_descent_v4_10000, 1, 75__ca-GrQc_uniform_1-1000_const_0_8.py', 'tdg_iter_descent_v4_10000, 1, 75__ca-GrQc_uniform_1-1000_uniform_0_75-1.py', 'tdg_iter_descent_v4_10000, 1, 75__facebook_combined_uniform_1-1000_const_0_8.py', 'tdg_iter_descent_v4_10000, 1, 75__facebook_combined_uniform_1-1000_uniform_0_75-1.py', 'tdg_iter_descent_v4_10000, 1, 75__p2p-Gnutella06_uniform_1-1000_const_0_8.py', 'tdg_iter_descent_v4_10000, 1, 75__p2p-Gnutella06_uniform_1-1000_uniform_0_75-1.py', 'tdg_iter_descent_v4_10000, 1, 75__p2p-Gnutella08_uniform_1-1000_uniform_0_75-1.py', 'tdg_iter_descent_v4_10000, 1, 75__p2p-Gnutella08_uniform_1-1000_const_0_8.py', 'tdg_iter_descent_v4_10000, 1, 75__Wiki-Vote_uniform_1-1000_const_0_8.py', 'tdg_iter_descent_v4_10000, 1, 75__Wiki-Vote_uniform_1-1000_uniform_0_75-1.py', 'tdg_iter_descent_v4_10000, 1, 100__ca-GrQc_uniform_1-1000_const_0_8.py', 'tdg_iter_descent_v4_10000, 1, 100__ca-GrQc_uniform_1-1000_uniform_0_75-1.py', 'tdg_iter_descent_v4_10000, 1, 100__facebook_combined_uniform_1-1000_const_0_8.py', 'tdg_iter_descent_v4_10000, 1, 100__facebook_combined_uniform_1-1000_uniform_0_75-1.py', 'tdg_iter_descent_v4_10000, 1, 100__p2p-Gnutella06_uniform_1-1000_const_0_8.py', 'tdg_iter_descent_v4_10000, 1, 100__p2p-Gnutella06_uniform_1-1000_uniform_0_75-1.py', 'tdg_iter_descent_v4_10000, 1, 100__p2p-Gnutella08_uniform_1-1000_uniform_0_75-1.py', 'tdg_iter_descent_v4_10000, 1, 100__p2p-Gnutella08_uniform_1-1000_const_0_8.py', 'tdg_iter_descent_v4_10000, 1, 100__Wiki-Vote_uniform_1-1000_const_0_8.py', 'tdg_iter_descent_v4_10000, 1, 100__Wiki-Vote_uniform_1-1000_uniform_0_75-1.py', 'tdg_iter_descent_v4_10000, 1, 125__ca-GrQc_uniform_1-1000_const_0_8.py', 'tdg_iter_descent_v4_10000, 1, 125__ca-GrQc_uniform_1-1000_uniform_0_75-1.py', 'tdg_iter_descent_v4_10000, 1, 125__facebook_combined_uniform_1-1000_const_0_8.py', 'tdg_iter_descent_v4_10000, 1, 125__facebook_combined_uniform_1-1000_uniform_0_75-1.py', 'tdg_iter_descent_v4_10000, 1, 125__p2p-Gnutella06_uniform_1-1000_const_0_8.py', 'tdg_iter_descent_v4_10000, 1, 125__p2p-Gnutella06_uniform_1-1000_uniform_0_75-1.py', 'tdg_iter_descent_v4_10000, 1, 125__p2p-Gnutella08_uniform_1-1000_uniform_0_75-1.py', 'tdg_iter_descent_v4_10000, 1, 125__p2p-Gnutella08_uniform_1-1000_const_0_8.py', 'tdg_iter_descent_v4_10000, 1, 125__Wiki-Vote_uniform_1-1000_const_0_8.py', 'tdg_iter_descent_v4_10000, 1, 125__Wiki-Vote_uniform_1-1000_uniform_0_75-1.py']

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
