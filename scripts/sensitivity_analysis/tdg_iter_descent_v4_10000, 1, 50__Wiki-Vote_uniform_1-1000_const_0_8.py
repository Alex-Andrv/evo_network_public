
task_name = "Wiki-Vote&uniform_1-1000&const_0.8"
solver_name, run_script = ("tdg_iter_descent_v4[10000, 1, 50]", lambda problem, seed: problem.solve_using_tdg_and_then_iter_descent_v4(None, None, tss_stop_criteria.by_iteration_count(10000), 1, 50, seed))
experiment_series = "sensitivity_analysis"

import sys
sys.path.append('/Users/aandreev/PycharmProjects/evo_network_public')
import src.tss_stop_criteria
from src.dltm import read_dltm
from src.tss import TSSProblem

if __name__ == '__main__':
    tss_name, dltm = task_name, read_dltm(f'generated_graph/large/{task_name}')
    tss = TSSProblem(dltm, dltm.nodes_count() * 0.75)
    with open(f'results/sensitivity_analysis/{solver_name}_{task_name}.csv', 'w') as f:
        f.write(f"{solver_name}\n")
        f.write("time,|TSS|,sol\n")
        for i in range(20):
            solution, metadata = run_script(tss, i)
            print(
                f'TSS instance {tss_name} ({{tss.nodes_count()}} nodes, {{tss.edges_count()}} edges) '
                f'solved using {solver_name}: time={{metadata["time"]}}, ts_size={{len(solution)}}, ts={{solution}}'
            )
            f.write(f'{{metadata["time"]}},{{len(solution)}},{{ "&".join(map(str, solution)) }}\n')
            f.flush()
