import json
import pickle
import networkx as nx
import argparse
import os
from collections import Counter
import tqdm

Too_big_community = []


# Perform Louvain clustering with binary search on resolution to find ~9–10 communities
def dynamic_resolution(G: nx.Graph, loose_to_8=False):
    l, r = 0.0, 5.0
    possible_number_cate = [9, 10]
    if loose_to_8:
        possible_number_cate.append(8)

    while l < r and (r - l) > 1e-2:
        mid = (l + r) / 2
        lc_list = nx.community.louvain_communities(G, resolution=mid, seed=666)
        if len(lc_list) > 10:
            r = mid
        elif len(lc_list) in possible_number_cate:
            break
        else:
            l = mid
    return lc_list


def main(args):
    if args.resume and os.path.exists(args.output_file):
        # Resume from previously clustered graph
        with open(args.output_file, "rb") as f:
            G_concept = pickle.load(f)
        print("Resuming from previous run...")
        cluster_list = sorted(
            [k for k, v in Counter([G_concept.nodes[n]["cluster"] for n in G_concept.nodes()]).items() if v > 1],
            key=len
        )
    else:
        # Load the precomputed graph with embeddings
        with open(args.input_file, "rb") as f:
            G_readin = pickle.load(f)

        G_concept = nx.DiGraph()
        G_concept.update(G_readin)
        del G_readin

        # First-level clustering using Louvain with dynamic resolution
        lc_list = dynamic_resolution(G_concept)
        print(f"The first clustering makes {len(lc_list)} clusters.")
        if len(lc_list) > 10:
            raise ValueError("Initial clustering exceeds expected cluster count.")

        # Assign cluster labels to nodes
        cluster_list = []
        for i, nodes in enumerate(lc_list):
            cluster_id = str(i)
            cluster_list.append(cluster_id)
            for node in nodes:
                G_concept.nodes[node]["cluster"] = cluster_id

    cluster_list = sorted(set(cluster_list))
    num_clusters = args.num_clusters

    # Lists to track clusters that need refinement
    community_too_big_list = []
    community_too_small_list = []

    # Recursive function to split subgraphs into smaller clusters
    def recursive_clustering(c_id: str, l_resolution: float):
        subset_nodes = [n for n in G_concept.nodes() if G_concept.nodes[n]["cluster"] == c_id]

        if 1 < len(subset_nodes) <= num_clusters:
            for i, node in enumerate(subset_nodes):
                G_concept.nodes[node]["cluster"] = f"{c_id}-{i}"
            return
        elif len(subset_nodes) == 1:
            return

        G_sub = G_concept.subgraph(subset_nodes).copy()
        cur_lc_list = dynamic_resolution(G_sub, loose_to_8=True)
        cur_lc_map = {str(i): list(cur_lc_list[i]) for i in range(len(cur_lc_list))}

        unique_clusters = list(cur_lc_map.keys())
        if len(unique_clusters) == 1:
            community_too_big_list.append(c_id)
            return
        elif len(unique_clusters) > num_clusters:
            community_too_small_list.append(c_id)
            return

        for cc_id, nodes in cur_lc_map.items():
            for node in nodes:
                G_concept.nodes[node]["cluster"] = f"{c_id}-{cc_id}"

        for cc_id in unique_clusters:
            recursive_clustering(f"{c_id}-{cc_id}", l_resolution)

    # First recursive partitioning round
    print("Start recursive clustering...")
    for c_id in tqdm.tqdm(cluster_list):
        recursive_clustering(c_id, l_resolution=0.25)

    # Refine clusters that are too small
    print(f"Refining small clusters: {len(community_too_small_list)} subgraphs")
    cur_small = community_too_small_list.copy()
    community_too_small_list.clear()
    for c_id in tqdm.tqdm(cur_small):
        recursive_clustering(c_id, l_resolution=0.3)

    # Refine clusters that are too big
    print(f"Refining large clusters: {len(community_too_big_list)} subgraphs")
    cur_big = community_too_big_list.copy()
    community_too_big_list.clear()
    for c_id in tqdm.tqdm(cur_big):
        recursive_clustering(c_id, l_resolution=1.0)

    print("Too big community:", community_too_big_list)
    print("Too small community:", community_too_small_list)

    # Save the graph with final hierarchical clustering
    with open(args.output_file, "wb") as f:
        pickle.dump(G_concept, f)
    print(f"Finished. Output written to: {args.output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Graph partitioning and hierarchical clustering.")
    parser.add_argument('--input_file', type=str, default="", help="Path to the input graph pickle file.")
    parser.add_argument('--output_file', type=str, default="", help="Path to save the clustered graph.")
    parser.add_argument('--num_clusters', type=int, default=10, help="Target number of clusters.")
    parser.add_argument('--resume', action='store_true', help="Resume from previously saved output.")

    args = parser.parse_args()

    # Example arguments (can be overwritten by command line)
    args.input_file = "models/hoip/embedding/graph_ossi.pkl"
    args.output_file = "models/hoip/embedding/graph_ossi_louvain.pkl"
    args.num_clusters = 10
    args.resume = False

    main(args)
