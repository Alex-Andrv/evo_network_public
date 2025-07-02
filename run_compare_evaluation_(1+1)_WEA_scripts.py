import os
import subprocess

scripts_dir = "scripts/evaluation_(1+1)_WEA"
working_dir = "/Users/aandreev/PycharmProjects/evo_network_public"

script_files = ['tdg_ca-GrQc_uniform_1-1000_const_0_8.py', 'tdg_ca-GrQc_uniform_1-1000_uniform_0_75-1.py', 'tdg_facebook_combined_uniform_1-1000_const_0_8.py', 'tdg_facebook_combined_uniform_1-1000_uniform_0_75-1.py', 'tdg_p2p-Gnutella06_uniform_1-1000_const_0_8.py', 'tdg_p2p-Gnutella06_uniform_1-1000_uniform_0_75-1.py', 'tdg_p2p-Gnutella08_uniform_1-1000_uniform_0_75-1.py', 'tdg_p2p-Gnutella08_uniform_1-1000_const_0_8.py', 'tdg_Wiki-Vote_uniform_1-1000_const_0_8.py', 'tdg_Wiki-Vote_uniform_1-1000_uniform_0_75-1.py', 'tdg_1+1_10000__ca-GrQc_uniform_1-1000_const_0_8.py', 'tdg_1+1_10000__ca-GrQc_uniform_1-1000_uniform_0_75-1.py', 'tdg_1+1_10000__facebook_combined_uniform_1-1000_const_0_8.py', 'tdg_1+1_10000__facebook_combined_uniform_1-1000_uniform_0_75-1.py', 'tdg_1+1_10000__p2p-Gnutella06_uniform_1-1000_const_0_8.py', 'tdg_1+1_10000__p2p-Gnutella06_uniform_1-1000_uniform_0_75-1.py', 'tdg_1+1_10000__p2p-Gnutella08_uniform_1-1000_uniform_0_75-1.py', 'tdg_1+1_10000__p2p-Gnutella08_uniform_1-1000_const_0_8.py', 'tdg_1+1_10000__Wiki-Vote_uniform_1-1000_const_0_8.py', 'tdg_1+1_10000__Wiki-Vote_uniform_1-1000_uniform_0_75-1.py', 'tdg_doerr1+1_3;10000__ca-GrQc_uniform_1-1000_const_0_8.py', 'tdg_doerr1+1_3;10000__ca-GrQc_uniform_1-1000_uniform_0_75-1.py', 'tdg_doerr1+1_3;10000__facebook_combined_uniform_1-1000_const_0_8.py', 'tdg_doerr1+1_3;10000__facebook_combined_uniform_1-1000_uniform_0_75-1.py', 'tdg_doerr1+1_3;10000__p2p-Gnutella06_uniform_1-1000_const_0_8.py', 'tdg_doerr1+1_3;10000__p2p-Gnutella06_uniform_1-1000_uniform_0_75-1.py', 'tdg_doerr1+1_3;10000__p2p-Gnutella08_uniform_1-1000_uniform_0_75-1.py', 'tdg_doerr1+1_3;10000__p2p-Gnutella08_uniform_1-1000_const_0_8.py', 'tdg_doerr1+1_3;10000__Wiki-Vote_uniform_1-1000_const_0_8.py', 'tdg_doerr1+1_3;10000__Wiki-Vote_uniform_1-1000_uniform_0_75-1.py', 'tdg_iter_descent_10000, 1__ca-GrQc_uniform_1-1000_const_0_8.py', 'tdg_iter_descent_10000, 1__ca-GrQc_uniform_1-1000_uniform_0_75-1.py', 'tdg_iter_descent_10000, 1__facebook_combined_uniform_1-1000_const_0_8.py', 'tdg_iter_descent_10000, 1__facebook_combined_uniform_1-1000_uniform_0_75-1.py', 'tdg_iter_descent_10000, 1__p2p-Gnutella06_uniform_1-1000_const_0_8.py', 'tdg_iter_descent_10000, 1__p2p-Gnutella06_uniform_1-1000_uniform_0_75-1.py', 'tdg_iter_descent_10000, 1__p2p-Gnutella08_uniform_1-1000_uniform_0_75-1.py', 'tdg_iter_descent_10000, 1__p2p-Gnutella08_uniform_1-1000_const_0_8.py', 'tdg_iter_descent_10000, 1__Wiki-Vote_uniform_1-1000_const_0_8.py', 'tdg_iter_descent_10000, 1__Wiki-Vote_uniform_1-1000_uniform_0_75-1.py', 'tdg_iter_descent_v2_10000, 1__ca-GrQc_uniform_1-1000_const_0_8.py', 'tdg_iter_descent_v2_10000, 1__ca-GrQc_uniform_1-1000_uniform_0_75-1.py', 'tdg_iter_descent_v2_10000, 1__facebook_combined_uniform_1-1000_const_0_8.py', 'tdg_iter_descent_v2_10000, 1__facebook_combined_uniform_1-1000_uniform_0_75-1.py', 'tdg_iter_descent_v2_10000, 1__p2p-Gnutella06_uniform_1-1000_const_0_8.py', 'tdg_iter_descent_v2_10000, 1__p2p-Gnutella06_uniform_1-1000_uniform_0_75-1.py', 'tdg_iter_descent_v2_10000, 1__p2p-Gnutella08_uniform_1-1000_uniform_0_75-1.py', 'tdg_iter_descent_v2_10000, 1__p2p-Gnutella08_uniform_1-1000_const_0_8.py', 'tdg_iter_descent_v2_10000, 1__Wiki-Vote_uniform_1-1000_const_0_8.py', 'tdg_iter_descent_v2_10000, 1__Wiki-Vote_uniform_1-1000_uniform_0_75-1.py', 'tdg_iter_descent_v3_10000, 1__ca-GrQc_uniform_1-1000_const_0_8.py', 'tdg_iter_descent_v3_10000, 1__ca-GrQc_uniform_1-1000_uniform_0_75-1.py', 'tdg_iter_descent_v3_10000, 1__facebook_combined_uniform_1-1000_const_0_8.py', 'tdg_iter_descent_v3_10000, 1__facebook_combined_uniform_1-1000_uniform_0_75-1.py', 'tdg_iter_descent_v3_10000, 1__p2p-Gnutella06_uniform_1-1000_const_0_8.py', 'tdg_iter_descent_v3_10000, 1__p2p-Gnutella06_uniform_1-1000_uniform_0_75-1.py', 'tdg_iter_descent_v3_10000, 1__p2p-Gnutella08_uniform_1-1000_uniform_0_75-1.py', 'tdg_iter_descent_v3_10000, 1__p2p-Gnutella08_uniform_1-1000_const_0_8.py', 'tdg_iter_descent_v3_10000, 1__Wiki-Vote_uniform_1-1000_const_0_8.py', 'tdg_iter_descent_v3_10000, 1__Wiki-Vote_uniform_1-1000_uniform_0_75-1.py', 'tdg_genetic_10000, 2, 4, 4__ca-GrQc_uniform_1-1000_const_0_8.py', 'tdg_genetic_10000, 2, 4, 4__ca-GrQc_uniform_1-1000_uniform_0_75-1.py', 'tdg_genetic_10000, 2, 4, 4__facebook_combined_uniform_1-1000_const_0_8.py', 'tdg_genetic_10000, 2, 4, 4__facebook_combined_uniform_1-1000_uniform_0_75-1.py', 'tdg_genetic_10000, 2, 4, 4__p2p-Gnutella06_uniform_1-1000_const_0_8.py', 'tdg_genetic_10000, 2, 4, 4__p2p-Gnutella06_uniform_1-1000_uniform_0_75-1.py', 'tdg_genetic_10000, 2, 4, 4__p2p-Gnutella08_uniform_1-1000_uniform_0_75-1.py', 'tdg_genetic_10000, 2, 4, 4__p2p-Gnutella08_uniform_1-1000_const_0_8.py', 'tdg_genetic_10000, 2, 4, 4__Wiki-Vote_uniform_1-1000_const_0_8.py', 'tdg_genetic_10000, 2, 4, 4__Wiki-Vote_uniform_1-1000_uniform_0_75-1.py']

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
