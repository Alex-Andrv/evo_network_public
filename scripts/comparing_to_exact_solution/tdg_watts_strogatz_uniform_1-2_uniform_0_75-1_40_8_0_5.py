
task_name = "watts_strogatz&uniform_1-2&uniform_0.75-1&40&8&0.5"
solver_name, run_script = ("tdg", lambda problem, seed: problem.solve_using_tdg(None, None))
experiment_series = "comparing_to_exact_solution"

import sys
sys.path.append('/Users/aandreev/PycharmProjects/evo_network_public')
import src.tss_stop_criteria
from src.dltm import read_dltm
from src.tss import TSSProblem

if __name__ == '__main__':
    tss_name, dltm = task_name, read_dltm(f'/Users/aandreev/PycharmProjects/evo_network_public/generated_graph/small/{task_name}')
    tss = TSSProblem(dltm, dltm.nodes_count() * 0.75)
    with open(f'/Users/aandreev/PycharmProjects/evo_network_public/results_csv/comparing_to_exact_solution/{solver_name}_{task_name}.csv', 'w') as f:
        f.write(f"{solver_name}\n")
        f.write("time,|TSS|,sol\n")
        for i in range(20):
            solution, metadata = run_script(tss, i)
            print(
                f'TSS instance {tss_name} ({tss.nodes_count()} nodes, {tss.edges_count()} edges) '
                f'solved using {solver_name}: time={{metadata["time"]}}, ts_size={{len(solution)}}, ts={{solution}}'
            )
            f.write(f'{{metadata["time"]}},{{len(solution)}},{{ "&".join(map(str, solution)) }}\n')
            f.flush()
