import os
seria = "seria_4_0.6"
run_script = f"run_script_{seria}.sh"


if os.path.exists(run_script):
   os.remove(run_script)

import shutil

if os.path.exists(f'generate/{seria}'):
   shutil.rmtree(f'generate/{seria}')

if os.path.exists(f'alex_exp/{seria}'):
   shutil.rmtree(f'alex_exp/{seria}')

os.mkdir(f'generate/{seria}')
os.mkdir(f'alex_exp/{seria}')

run_script_template = """
    sbatch --cpus-per-task=2 --mem=20G -p as --time=20:00:00 -w orthrus-2 --qos=high_cpu --qos=high_mem --qos=unlim_cpu "generate/{}/{}"
"""

python_script_template_head = """

task_name = "{}"
solver_name, run_script = ("{}",{})
seria = "{}"

"""

python_script_template_body = """

import os 
os.chdir('/nfs/home/aandreev/evo_network/')
import sys 
sys.path.append('/nfs/home/aandreev/evo_network/')
import tss_stop_criteria
from dltm import read_dltm
from tss import TSSProblem

if __name__ == '__main__':

    tss_name, dltm = task_name, read_dltm(f'/nfs/home/aandreev/evo_network/generated_graph/{seria}/{task_name}')
    tss = TSSProblem(dltm, dltm.nodes_count() * 0.6)
    with open(f'alex_exp/{seria}/{solver_name}&{task_name}.csv', 'w') as f:
        f.write(f"{solver_name}\\n")
        f.write(f"time,|TSS|,sol\\n")
        for i in range(20):
            solution, metadata = run_script(tss, i)
            print('TSS instance {} ({} nodes, {} edges) solved using {}:  time={}, ts_size={}, ts={}'
                      .format(tss_name, tss.nodes_count(), tss.edges_count(), solver_name, metadata['time'],
                              len(solution), solution))
            f.write(str(metadata['time']))
            f.write(",")
            f.write(str(len(solution)))
            f.write(",")
            f.write(str('&'.join(map(str, solution))))
            f.write('\\n')
            f.flush()
"""

sh_script_template = """#!/bin/bash -i
source activate evo_network
rm "/nfs/home/aandreev/evo_network/generate/{}"
chmod +x "/nfs/home/aandreev/evo_network/generate/{}"
python "/nfs/home/aandreev/evo_network/generate/{}" >> "/nfs/home/aandreev/evo_network/generate/{}" 2>&1
"""

if __name__ == '__main__':
 solvers = [
     ('tdg', "lambda problem, seed: problem.solve_using_tdg(None, None)"),
     ('tdg_iter_descent_IMP[5000, 1]',
      "lambda problem, seed: problem.solve_using_tdg_and_then_iter_descent_IMP(None, None, tss_stop_criteria.by_iteration_count(5000), 1, seed)"),
  ]
 
 tasks = [
    "ca-GrQc&uniform_1-1000&const_0.8",
    "ca-GrQc&uniform_1-1000&uniform_0.75-1",
    "facebook_combined&uniform_1-1000&const_0.8",
    "facebook_combined&uniform_1-1000&uniform_0.75-1",
    "p2p-Gnutella06&uniform_1-1000&const_0.8", 
    "p2p-Gnutella06&uniform_1-1000&uniform_0.75-1",
    "p2p-Gnutella08&uniform_1-1000&uniform_0.75-1", 
    "p2p-Gnutella08&uniform_1-1000&const_0.8",
    "Wiki-Vote&uniform_1-1000&const_0.8",
    "Wiki-Vote&uniform_1-1000&uniform_0.75-1"
 ]
 for solver_name, run_scrip in solvers:
     for task in tasks:
         full_name = f"{solver_name}&{task}"
         with open(f"generate/{seria}/{full_name}.py", "w") as python_script:
             script = f"""
             {python_script_template_head.format(task, solver_name, run_scrip, seria)}
             {python_script_template_body}
             """
             python_script.write(script)
         with open(f"generate/{seria}/{full_name}.sh", "w") as python_script:
             python_script.write(sh_script_template.format(f"{seria}/{full_name}.txt", f"{seria}/{full_name}.py", f"{seria}/{full_name}.py", f"{seria}/{full_name}.txt"))
         with open(run_script, 'a') as run_script_file:
             run_script_file.write(run_script_template.format(seria, f"{full_name}.sh"))
         


