import json
import pickle
import random
import networkx as nx
import pandas as pd


def map_index(index_types: list, concept_type: str):
    """
    Maps cluster IDs to original concept ontology IDs.
    Saves the result as a CSV (for tabular viewing) and JSON (for fast access in code).
    """
    for index_type in index_types:
        do_random = False  # Set to True if you want randomized cluster assignments (not used here)

        # Load the clustered graph
        graph_path = f"models/{concept_type}/embedding/graph_{index_type}_louvain.pkl"
        with open(graph_path, "rb") as f:
            G_readin = pickle.load(f)

        G = nx.DiGraph()
        G.update(G_readin)
        del G_readin  # Free memory

        id_to_cluster = {}
        labels = []

        for node in G.nodes():
            onto_id = G.nodes[node]["onto_id"]
            label = str(G.nodes[node]["label"])

            # Format full ontology URI based on ontology prefix
            if onto_id.startswith("GO_") or onto_id.startswith("HP_"):
                concept_uri = f"http://purl.obolibrary.org/obo/{onto_id}"
            else:
                concept_uri = f"http://purl.bioontology.org/ontology/HOIP/{onto_id}"

            # Convert hierarchical cluster ID to list of ints (e.g., "0-2-1" -> [0,2,1])
            cluster_path = [int(part) for part in G.nodes[node]["cluster"].split("-")]

            id_to_cluster[concept_uri] = cluster_path
            labels.append(label)

        # Optional random shuffle (disabled)
        concept_ids = list(id_to_cluster.keys())
        cluster_ids = list(id_to_cluster.values())
        if do_random:
            random.shuffle(cluster_ids)

        # Rebuild mapping (ensuring alignment)
        id_to_cluster = {cid: sid for cid, sid in zip(concept_ids, cluster_ids)}

        # Save to CSV for inspection
        df = pd.DataFrame({
            "label": labels,
            "entity_id": concept_ids,
            "search_id": cluster_ids
        })
        csv_path = f"../data/search_index/{concept_type}/label_id_{index_type}.csv"
        df.to_csv(csv_path, sep='\t', index=False)

        # Save to JSON for indexing
        json_path = f"../data/search_index/{concept_type}/id_{index_type}.json"
        with open(json_path, "w") as f_out:
            json.dump(id_to_cluster, f_out)


if __name__ == '__main__':
    index_types = ["osi", "ssi", "ossi"]  # Variants of graph types or feature sets
    concept_type = "hpa"  # Can be "hpa", "hoip", etc.
    map_index(index_types, concept_type)
