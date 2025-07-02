import click
from pathlib import Path
from src.dltm import DLTM, write_dltm


@click.command()
@click.option(
    "--small-dir",
    default="generated_graph/small",
    show_default=True,
    help="Directory to save small graphs (BA, WS)",
)
@click.option(
    "--large-dir",
    default="generated_graph/large",
    show_default=True,
    help="Directory to save large graphs (Wiki-Vote, p2p, Facebook, ca-GrQc)",
)
@click.option(
    "--seed",
    default=123,
    show_default=True,
    help="Seed for graph and weight generation",
)
def generate_datasets(small_dir, large_dir, seed):
    """
    Generate datasets for the first (small graphs) and second (large graphs) experiment series
    with fixed parameters for reproducibility.
    """

    # Ensure output directories exist
    Path(small_dir).mkdir(parents=True, exist_ok=True)
    Path(large_dir).mkdir(parents=True, exist_ok=True)

    print(f"[+] Generating small graphs in {small_dir} ...")
    # First series: small graphs (BA and WS)

    # BA 50 4
    write_dltm(
        DLTM()
        .generate_barabasi_albert_graph(50, 4, seed=seed)
        .generate_uniformly_random_influences(1, 5, seed=seed)
        .generate_uniformly_random_thresholds(0.75, 1, seed=seed),
        f"{small_dir}/barabasi_albert&uniform_1-5&uniform_0.75-1&50&4",
    )

    write_dltm(
        DLTM()
        .generate_barabasi_albert_graph(50, 4, seed=seed)
        .generate_uniformly_random_influences(1, 5, seed=seed)
        .generate_proportional_thresholds(0.8),
        f"{small_dir}/barabasi_albert&uniform_1-5&const_0.8&50&4",
    )

    # WS 40 8 0.5
    write_dltm(
        DLTM()
        .generate_watts_strogatz_graph(40, 8, 0.5, seed=seed)
        .generate_uniformly_random_influences(1, 2, seed=seed)
        .generate_uniformly_random_thresholds(0.75, 1, seed=seed),
        f"{small_dir}/watts_strogatz&uniform_1-2&uniform_0.75-1&40&8&0.5",
    )

    write_dltm(
        DLTM()
        .generate_watts_strogatz_graph(40, 8, 0.5, seed=seed)
        .generate_uniformly_random_influences(1, 2, seed=seed)
        .generate_proportional_thresholds(0.8),
        f"{small_dir}/watts_strogatz&uniform_1-2&const_0.8&40&8&0.5",
    )

    print(f"[+] Generating large graphs in {large_dir} ...")
    # Second series: large graphs from SNAP datasets
    big_graphs = [
        ("samples/Wiki-Vote.txt", True, "Wiki-Vote"),
        ("samples/p2p-Gnutella06.txt", True, "p2p-Gnutella06"),
        ("samples/p2p-Gnutella08.txt", True, "p2p-Gnutella08"),
        ("samples/facebook_combined.txt", False, "facebook_combined"),
        ("samples/ca-GrQc.txt", True, "ca-GrQc"),
    ]

    for path, directed, name in big_graphs:
        dltm = DLTM().read_graph_as_edge_list(path, directed)
        # Constant threshold
        write_dltm(
            dltm.read_graph_as_edge_list(path, directed)
            .generate_uniformly_random_influences(1, 1000, seed=seed)
            .generate_proportional_thresholds(0.8),
            f"{large_dir}/{name}&uniform_1-1000&const_0.8",
        )

        # Uniform random threshold
        write_dltm(
            dltm.read_graph_as_edge_list(path, directed)
            .generate_uniformly_random_influences(1, 1000, seed=seed)
            .generate_uniformly_random_thresholds(0.75, 1, seed=seed),
            f"{large_dir}/{name}&uniform_1-1000&uniform_0.75-1",
        )

    print("[+] Generation completed successfully!")


if __name__ == "__main__":
    generate_datasets()
