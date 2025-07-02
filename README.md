## Preparing

1) Create the virtual environment named evo_network:

```
python3.12 -m venv evo_network
```

2) Activate the environment:

```
source evo_network/bin/activate
```

3) Upgrade pip:

```
pip install --upgrade pip
```

4) Install dependencies:

```
pip install -r requirements.txt
```

## Reproducing the Experiments
Once the datasets are generated, run the provided experiment launchers to reproduce the results.

The launchers are Python scripts that will execute all experiments sequentially and log outputs:

### Comparison to Exact Solution

```
python run_compare_exact_scripts.py
```

### Evaluation of (1+1)-WEA on Large Graphs

```
python run_compare_evaluation_(1+1)_WEA_scripts.py
```

### Sensitivity Analysis

```
python run_compare_sensitivity_analysis_scripts.py
```
Each launcher will generate logs next to the experiment scripts for inspection.

## Graph generation

To generate the datasets used in the experiments, run the provided script from the command line:

```bash
python generate_datasets.py
```

This will generate:

✅ **Small graphs** in the directory:
`generated_graph/small/` (synthetic networks: Barabási-Albert and Watts-Strogatz).

✅ **Large graphs** in the directory:
`generated_graph/large/` (real networks: Wiki-Vote, p2p-Gnutella, Facebook, ca-GrQc).

You can customize the output directories or the random seed using the script options:

```bash
python generate_datasets.py --small-dir <small_dir_path> --large-dir <large_dir_path> --seed 123
```

### Graph usage in the paper

🔹 **Small graphs**:
Used for analysis on synthetic Barabási-Albert (BA) and Watts-Strogatz (WS) models to study the impact of different parameters on small, controlled networks (see subsection **4.2 Comparing to Exact Solution**).

🔹 **Large graphs**:
Used to evaluate the performance and applicability on real networks from SNAP (Wiki-Vote, p2p, Facebook, ca-GrQc), demonstrating the scalability and efficiency of the approach on large graphs, as well as conducting sensitivity analysis of the method (see subsections **4.4 Evaluation of (1+1)-WEA for Solving TSS** and **4.5 Sensitivity Analysis**).