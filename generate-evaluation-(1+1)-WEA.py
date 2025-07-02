import os
import shutil
import argparse

parser = argparse.ArgumentParser(
    description="Generate experiment scripts and Python launcher."
)
parser.add_argument(
    "--graph-dir",
    type=str,
    default="generated_graph/large",
    help="Path to the directory with generated graphs relative to working dir (default: generated_graph/large)",
)
args = parser.parse_args()

experiment_series = "evaluation_(1+1)_WEA"
working_dir = os.getcwd()
graph_dir = args.graph_dir

scripts_dir = f"scripts/{experiment_series}"
results_dir = f"results/{experiment_series}"

# Clean old scripts and results directories
for path in [scripts_dir, results_dir]:
    if os.path.isdir(path):
        shutil.rmtree(path)
os.makedirs(scripts_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

python_script_template_head = """
task_name = "{}"
solver_name, run_script = ("{}", {})
experiment_series = "{}"
"""

python_script_template_body = (
    """
import sys
sys.path.append('{working_dir}')
import src.tss_stop_criteria
from src.dltm import read_dltm
from src.tss import TSSProblem

if __name__ == '__main__':
    tss_name, dltm = task_name, read_dltm(f'{graph_dir}/{task_name}')
    tss = TSSProblem(dltm, dltm.nodes_count() * 0.75)
    with open(f'{results_dir}/{solver_name}_{task_name}.csv', 'w') as f:
        f.write(f"{solver_name}\\n")
        f.write("time,|TSS|,sol\\n")
        for i in range(20):
            solution, metadata = run_script(tss, i)
            print(
                f'TSS instance {tss_name} ({{tss.nodes_count()}} nodes, {{tss.edges_count()}} edges) '
                f'solved using {solver_name}: time={{metadata["time"]}}, ts_size={{len(solution)}}, ts={{solution}}'
            )
            f.write(f'{{metadata["time"]}},{{len(solution)}},{{ "&".join(map(str, solution)) }}\\n')
            f.flush()
""".replace(
        "{working_dir}", working_dir
    )
    .replace("{results_dir}", results_dir)
    .replace("{graph_dir}", graph_dir)
)


solvers = [
    ("tdg", "lambda problem, seed: problem.solve_using_tdg(None, None)"),
    (
        "tdg_1+1[10000]",
        "lambda problem, seed: problem.solve_using_tdg_and_then_1p1(None, None,tss_stop_criteria.by_iteration_count(10000), seed)",
    ),
    (
        "tdg_doerr1+1[3;10000]",
        "lambda problem, seed: problem.solve_using_tdg_and_then_doerr_1p1(None, None, 3,tss_stop_criteria.by_iteration_count(10000),seed)",
    ),
    (
        "tdg_iter_descent[10000, 1]",
        "lambda problem, seed: problem.solve_using_tdg_and_then_iter_descent(None, None, tss_stop_criteria.by_iteration_count(10000), 1, seed)",
    ),
    (
        "tdg_iter_descent_v2[10000, 1]",
        "lambda problem, seed: problem.solve_using_tdg_and_then_iter_descent_v2(None, None, tss_stop_criteria.by_iteration_count(10000), 1, seed)",
    ),
    (
        "tdg_iter_descent_v3[10000, 1]",
        "lambda problem, seed: problem.solve_using_tdg_and_then_iter_descent_v3(None, None, tss_stop_criteria.by_iteration_count(10000), 1, seed)",
    ),
    (
        "tdg_genetic[10000, 2, 4, 4]",
        "lambda problem, seed: problem.solve_using_custom_ga(2, 4, 4, tss_stop_criteria.by_iteration_count(10000), seed=seed)",
    ),
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
    "Wiki-Vote&uniform_1-1000&uniform_0.75-1",
]

generated_scripts = []

for solver_name, run_scrip in solvers:
    for task in tasks:
        full_name = f"{solver_name}&{task}"
        safe_name = (
            full_name.replace("&", "_")
            .replace("[", "_")
            .replace("]", "_")
            .replace(".", "_")
        )
        py_filename = f"{safe_name}.py"

        with open(os.path.join(scripts_dir, py_filename), "w") as py_script:
            script_content = (
                python_script_template_head.format(
                    task, solver_name, run_scrip, experiment_series
                )
                + python_script_template_body
            )
            py_script.write(script_content)

        generated_scripts.append(py_filename)

launcher_script = f"run_compare_{experiment_series}_scripts.py"
with open(launcher_script, "w") as launcher:
    launcher.write(
        f"""import os
import subprocess

scripts_dir = "{scripts_dir}"
working_dir = "{working_dir}"

script_files = {generated_scripts!r}

for script_name in script_files:
    script_path = os.path.join(scripts_dir, script_name)
    log_path = script_path.replace('.py', '.log')

    print(f"Running {{script_name}}...")

    with open(log_path, 'w') as log_file:
        process = subprocess.Popen(
            ['python', script_path],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=working_dir
        )
        retcode = process.wait()

    if retcode == 0:
        print(f"[DONE] {{script_name}} completed successfully.")
    else:
        print(f"[ERROR] {{script_name}} exited with code {{retcode}}. See {{log_path}} for details.")

print("All experiments completed.")
"""
    )

print(f"[INFO] Generated {len(generated_scripts)} experiment scripts in {scripts_dir}")
print(f"[INFO] Generated Python launcher script: {launcher_script}")
print(f"[INFO] To run all experiments, execute: python {launcher_script}")
