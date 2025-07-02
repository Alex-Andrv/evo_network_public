import os
import shutil
import argparse

parser = argparse.ArgumentParser(description="Generate experiment scripts.")
parser.add_argument(
    "--graph-dir",
    type=str,
    default="generated_graph/small",
    help="Path to the directory with generated graphs relative to working dir (default: generated_graph/small)",
)
args = parser.parse_args()

experiment_series = "comparing_to_exact_solution"
scripts_dir = f"scripts/{experiment_series}"

# Determine the absolute working directory
working_dir = os.getcwd()
graph_dir = args.graph_dir

# Clean old scripts directory
if os.path.exists(scripts_dir):
    shutil.rmtree(scripts_dir)
os.makedirs(scripts_dir, exist_ok=True)

# Create results directory
results_dir = f"results_csv/{experiment_series}"
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
    tss_name, dltm = task_name, read_dltm(f'{working_dir}/{graph_dir}/{task_name}')
    tss = TSSProblem(dltm, dltm.nodes_count() * 0.75)
    with open(f'{working_dir}/{results_dir}/{solver_name}_{task_name}.csv', 'w') as f:
        f.write(f"{solver_name}\\n")
        f.write("time,|TSS|,sol\\n")
        for i in range(20):
            solution, metadata = run_script(tss, i)
            print(
                f'TSS instance {tss_name} ({tss.nodes_count()} nodes, {tss.edges_count()} edges) '
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

if __name__ == "__main__":
    solvers = [
        ("exact_sat", "lambda problem, seed: problem.solve_using_sat()"),
        ("tdg", "lambda problem, seed: problem.solve_using_tdg(None, None)"),
        (
            "1+1[10000]",
            "lambda problem, seed: problem.solve_using_1p1(tss_stop_criteria.by_iteration_count(10000), seed=seed)",
        ),
        (
            "tdg_1+1[10000]",
            "lambda problem, seed: problem.solve_using_tdg_and_then_1p1(None, None, tss_stop_criteria.by_iteration_count(10000), seed=seed)",
        ),
    ]
    tasks = [
        "barabasi_albert&uniform_1-5&const_0.8&50&4",
        "watts_strogatz&uniform_1-2&const_0.8&40&8&0.5",
        "barabasi_albert&uniform_1-5&uniform_0.75-1&50&4",
        "watts_strogatz&uniform_1-2&uniform_0.75-1&40&8&0.5",
    ]

    generated_scripts = []

    for solver_name, run_scrip in solvers:
        for task in tasks:
            full_name = f"{solver_name}&{task}"
            safe_full_name = (
                full_name.replace("&", "_")
                .replace("[", "_")
                .replace("]", "_")
                .replace(".", "_")
            )
            py_filename = f"{safe_full_name}.py"

            # Generate the Python script for the experiment
            with open(os.path.join(scripts_dir, py_filename), "w") as py_script:
                script_content = (
                    python_script_template_head.format(
                        task, solver_name, run_scrip, experiment_series
                    )
                    + python_script_template_body
                )
                py_script.write(script_content)

            generated_scripts.append(py_filename)

    # Generate launcher script to run all experiments sequentially
    launcher_script = "run_compare_exact_scripts.py"
    with open(launcher_script, "w") as launcher:
        launcher.write(
            f"""import os
import subprocess

experiment_series = "{experiment_series}"
scripts_dir = "scripts/{{}}".format(experiment_series)
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

    print(
        f"[INFO] Generated {len(generated_scripts)} experiment scripts in {scripts_dir}"
    )
    print(f"[INFO] Generated launcher script: {launcher_script}")
    print(f"[INFO] To run all experiments, execute: python {launcher_script}")
