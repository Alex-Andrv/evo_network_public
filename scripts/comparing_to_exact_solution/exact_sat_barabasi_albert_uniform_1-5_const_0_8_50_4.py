
task_name = "barabasi_albert&uniform_1-5&const_0.8&50&4"
solver_name, run_script = ("exact_sat", lambda problem, seed: problem.solve_using_sat())
experiment_series = "comparing_to_exact_solution"

import sys
sys.path.append('/nfs/home/aandreev/evo_network')
import src.tss_stop_criteria
from src.dltm import read_dltm
from src.tss import TSSProblem

if __name__ == '__main__':
    tss_name, dltm = task_name, read_dltm(f'/nfs/home/aandreev/evo_network/generated_graph/{experiment_series}/{task_name}')
    tss = TSSProblem(dltm, dltm.nodes_count() * 0.75)
    with open(f'/nfs/home/aandreev/evo_network/results_csv/comparing_to_exact_solution/{solver_name}_{task_name}.csv', 'w') as f:
        f.write(f"{solver_name}\n")
        f.write("time,|TSS|,sol\n")
        for i in range(20):
            solution, metadata = run_script(tss, i)
            print(
                f'TSS instance {tss_name} ({tss.nodes_count()} nodes, {tss.edges_count()} edges) '
                f'solved using {solver_name}: time={metadata["time"]}, ts_size={len(solution)}, ts={solution}'
            )
            f.write(f'{metadata["time"]},{len(solution)},{ "&".join(map(str, solution)) }\n')
            f.flush()
