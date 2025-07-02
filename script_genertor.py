python_script_template_head = """

task_name = "{}"
solver_name, run_script = ("{}",{})

"""

python_script_template_body = """
import tss_stop_criteria
from dltm import read_dltm
from tss import TSSProblem

if __name__ == '__main__':

    tss_name, dltm = task_name, read_dltm(f'/nfs/home/aandreev/evo_network/generated_graph/{task_name}')
    tss = TSSProblem(dltm, dltm.nodes_count() * 0.75)
    with open(f'alex_exp/{solver_name}_{task_name}.csv', 'w') as f:
        f.write(f"{solver_name}\\n")
        f.write(f"time,|TSS|\\n")
        for i in range(20):
            solution, metadata = run_script(tss, i)
            print('TSS instance {} ({} nodes, {} edges) solved using {}:  time={}, ts_size={}, ts={}'
                      .format(tss_name, tss.nodes_count(), tss.edges_count(), solver_name, metadata['time'],
                              len(solution), solution))
            f.write(str(metadata['time']))
            f.write(",")
            f.write(str(len(solution)))
            f.write('\\n')
            f.flush()
"""

sh_script_template = """#!/bin/bash -i
source activate evo_network
chmod +x "/nfs/home/aandreev/evo_network/{}"
python "/nfs/home/aandreev/evo_network/{}" >> "{}" 2>&1
"""

if __name__ == '__main__':
 solvers = [
     ('tdg', "lambda problem, seed: problem.solve_using_tdg(None, None)"),
     ('tdg&1+1[10000]',
      "lambda problem, seed: problem.solve_using_tdg_and_then_1p1(None, None,tss_stop_criteria.by_iteration_count(10000), seed)"),
     ('tdg&doerr1+1[3;10000]',
      "lambda problem, seed: problem.solve_using_tdg_and_then_doerr_1p1(None, None, 3,tss_stop_criteria.by_iteration_count(10000),seed)"),
     ('tdg&iter_descent[10000, 1]',
      "lambda problem, seed: problem.solve_using_tdg_and_then_iter_descent(None, None, tss_stop_criteria.by_iteration_count(10000), 1, seed)"),
     ('tdg&iter_descent[10000, 2]',
      "lambda problem, seed: problem.solve_using_tdg_and_then_iter_descent(None, None, tss_stop_criteria.by_iteration_count(10000), 2, seed)"),
     ('tdg&iter_descent[10000, 5]',
      "lambda problem, seed: problem.solve_using_tdg_and_then_iter_descent(None, None, tss_stop_criteria.by_iteration_count(10000), 5, seed)"),
     ('tdg&iter_descent_v2[10000, 1]',
      "lambda problem, seed: problem.solve_using_tdg_and_then_iter_descent_v2(None, None, tss_stop_criteria.by_iteration_count(10000), 1, seed)"),
     ('tdg&iter_descent_v3[10000, 1]',
      "lambda problem, seed: problem.solve_using_tdg_and_then_iter_descent_v3(None, None, tss_stop_criteria.by_iteration_count(10000), 1, seed)"),
 ]
 tasks = [
     # "barabasi_albert_uniform_1-100_const_0.75",
     # "barabasi_albert_uniform_1-100_uniform_0.5-1",
     "facebook_combined_uniform_1-100_const_0.75",
     "facebook_combined_uniform_1-100_uniform_0.5-1",
     # "watts_strogatz_uniform_1-100_const_0.75",
     # "watts_strogatz_uniform_1-100_uniform_0.5-1",
     "Wiki-Vote_uniform_1-100_const_0.75",
     "Wiki-Vote_uniform_1-100_uniform_0.5-1",
     "ca-GrQc_uniform_1-100_const_0.75",
     "ca-GrQc_uniform_1-100_uniform_0.5-1",
     "p2p-Gnutella06_uniform_1-100_const_0.75",
     "p2p-Gnutella06_uniform_1-100_uniform_0.5-1",
     "p2p-Gnutella08_uniform_1-100_const_0.75",
     "p2p-Gnutella08_uniform_1-100_uniform_0.5-1"
 ]
 for solver_name, run_scrip in solvers:
     for task in tasks:
         with open(f"{solver_name}_{task}.py", "w") as python_script:
             script = f"""
             {python_script_template_head.format(task, solver_name, run_scrip)}
             {python_script_template_body}
             """
             python_script.write(script)
         with open(f"{solver_name}_{task}.sh", "w") as python_script:
             python_script.write(sh_script_template.format(f"{solver_name}_{task}.py", f"{solver_name}_{task}.py", f"{solver_name}_{task}.txt"))
