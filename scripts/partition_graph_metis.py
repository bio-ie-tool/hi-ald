import pickle
import networkx as nx
import pymetis
import argparse
import os
from multiprocessing import Process, Queue
from collections import Counter
from tqdm import tqdm


def partition_graph_worker(G: nx.Graph, num_partitions: int, queue: Queue):
    """
    Worker function for graph partitioning using PyMetis.
    Sends the partitioning result (cuts and membership) back via queue.
    """
    adj_list = [list(G.neighbors(node)) for node in G.nodes]
    cuts, membership = pymetis.part_graph(num_partitions, adjacency=adj_list)
    queue.put((cuts, membership))


def perform_partitioning_in_subprocess(G: nx.Graph, num_partitions: int):
    """
    Perform graph partitioning in a separate subprocess to avoid memory issues.
    """
    queue = Queue()
    process = Process(target=partition_graph_worker, args=(G, num_partitions, queue))
    process.start()
    process.join()
    return queue.get()


def recursive_clustering(c_id: str, G: nx.Graph, num_clusters: int, cluster_queue: Queue):
    """
    Recursively cluster a subgraph and update cluster labels in the main graph.
    """
    # Backup graph at each recursive step
    with open(args.output_file, "wb") as f:
        pickle.dump(G, f)

    subset_nodes = [node for node in G.nodes if G.nodes[node]["cluster"] == c_id]

    if 1 < len(subset_nodes) <= num_clusters:
        for cluster_idx, node in enumerate(subset_nodes):
            G.nodes[node]["cluster"] = f"{c_id}-{cluster_idx}"
        return
    elif len(subset_nodes) == 1:
        return

    # Extract and relabel the subgraph for partitioning
    G_sub = G.subgraph(subset_nodes).copy()
    mapping = {node: idx for idx, node in enumerate(G_sub.nodes())}
    G_sub = nx.relabel_nodes(G_sub, mapping)

    # Perform partitioning using PyMetis
    cuts, membership = perform_partitioning_in_subprocess(G_sub, num_clusters)

    if len(set(membership)) <= 1:
        return

    reverse_mapping = {v: k for k, v in mapping.items()}
    nodeid_to_cluster = {
        reverse_mapping[i]: f"{c_id}-{cluster_id}" for i, cluster_id in enumerate(membership)
    }

    for node, cluster in nodeid_to_cluster.items():
        G.nodes[node]["cluster"] = cluster

    # Add new clusters to the queue for further recursion
    for cluster_id in set(membership):
        next_cluster = f"{c_id}-{cluster_id}"
        cluster_queue.put(next_cluster)
        recursive_clustering(next_cluster, G, num_clusters, cluster_queue)


def main(args):
    """
    Main procedure for loading the graph, performing recursive clustering,
    and saving the clustered result.
    """
    if args.resume and os.path.exists(args.output_file):
        # Resume from previously clustered graph
        with open(args.output_file, "rb") as f:
            G_concept = pickle.load(f)
        print("Resuming from previous run...")
    else:
        # Initial clustering from raw graph
        with open(args.input_file, "rb") as f:
            G_concept = pickle.load(f)

        mapping = {node: idx for idx, node in enumerate(G_concept.nodes())}
        G_relabel = nx.relabel_nodes(G_concept, mapping)

        cuts, membership = perform_partitioning_in_subprocess(G_relabel, args.num_clusters)

        reverse_mapping = {v: k for k, v in mapping.items()}
        for i, cluster_id in enumerate(membership):
            original_node = reverse_mapping[i]
            G_concept.nodes[original_node]["cluster"] = str(cluster_id)

        with open(args.output_file, "wb") as f:
            pickle.dump(G_concept, f)
        return

    # Convert all cluster labels to string
    for node in G_concept.nodes():
        G_concept.nodes[node]["cluster"] = str(G_concept.nodes[node]["cluster"])

    # Prepare initial cluster list for recursion
    cluster_counter = Counter(G_concept.nodes[node]["cluster"] for node in G_concept.nodes())
    cluster_list = sorted([cid for cid, count in cluster_counter.items() if count > 1], key=len)

    cluster_queue = Queue()
    for c_id in cluster_list:
        cluster_queue.put(c_id)

    # Perform recursive clustering with progress bar
    with tqdm(total=len(cluster_list)) as p_bar:
        while not cluster_queue.empty():
            current_cluster = cluster_queue.get()
            recursive_clustering(current_cluster, G_concept, args.num_clusters, cluster_queue)
            p_bar.update(1)

    # Save final graph
    with open(args.output_file, "wb") as f:
        pickle.dump(G_concept, f)

    print(f"Finished. Output saved to: {args.output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Recursive graph partitioning using PyMetis.")
    parser.add_argument('--input_file', type=str, default="", help="Path to input graph (.pkl format).")
    parser.add_argument('--output_file', type=str, default="", help="Path to output graph (.pkl format).")
    parser.add_argument('--num_clusters', type=int, default=10, help="Target number of clusters.")
    parser.add_argument('--resume', action='store_true', help="Resume from existing output.")

    args = parser.parse_args()

    # Example arguments (can be overwritten by command line)
    args.input_file = "models/hoip/embedding/graph_ossi.pkl"
    args.output_file = "models/hoip/embedding/graph_ossi_louvain.pkl"
    args.num_clusters = 10
    args.resume = True

    main(args)
