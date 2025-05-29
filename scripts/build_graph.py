import json
import pickle
from typing import List, Tuple

import faiss
import networkx as nx
import numpy as np
from tqdm import tqdm


# ---------- Distance Functions ----------
class DistanceL2:
    def compare(self, a: np.ndarray, b: np.ndarray) -> float:
        diff = a - b
        return np.sum(diff ** 2)


# ---------- Approximate Nearest Neighbor Search ----------
class ApproximateNearestNeighborSearch:
    def __init__(self):
        self.index = None
        self.doc_ids = []
        self.doc_titles = []

    def make_index(self, vectors: np.ndarray, ids: List[str], titles: List[str]):
        dim = vectors.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(vectors)
        self.doc_ids = ids
        self.doc_titles = titles

    def search(self, queries: np.ndarray, top_k: int) -> Tuple[List[List[str]], List[List[float]]]:
        distances, indices = self.index.search(queries, top_k)
        scores = 1.0 / (distances + 1.0)
        results = [[self.doc_ids[i] for i in idx] for idx in indices]
        return results, scores


# ---------- Graph Construction ----------
def acquire_neighbors(node: str, candidates: List[str], max_neighbors: int,
                      graph: nx.Graph, dist_func=DistanceL2) -> Tuple[List[str], List[float]]:
    """Select a set of diverse and closest neighbors for a given node."""
    x_vec = np.array(graph.nodes[node]["vector"])
    c_vecs = [np.array(graph.nodes[c]["vector"]) for c in candidates]
    dist = dist_func()

    distances = [dist.compare(x_vec, v) for v in c_vecs]
    sorted_candidates = sorted(zip(candidates, distances), key=lambda x: x[1])

    neighbors, scores = [], []
    for cand, d in sorted_candidates:
        if len(neighbors) >= max_neighbors:
            break
        if all(dist.compare(np.array(graph.nodes[cand]["vector"]),
                            np.array(graph.nodes[n]["vector"])) > d for n in neighbors):
            neighbors.append(cand)
            scores.append(d)

    return neighbors, scores


def build_graph(concept_file: str, candidate_file: str, ontology_file_1: str, ontology_file_2: str, output_file: str,
                use_semantic_similarity: bool = True, use_ontology_structure: bool = False,
                L: int = 10, M: int = 10) -> nx.DiGraph:
    with open(concept_file, 'rb') as f:
        ids, vectors, labels = pickle.load(f)

    with open(candidate_file, 'r') as f:
        candidate_roots = json.load(f)

    # Filter concept nodes
    G = nx.DiGraph()
    concept_nodes = []
    for idx, vec, label in zip(ids, vectors, labels):
        root = "/".join(idx.split("/")[:-1])
        if root in candidate_roots and idx.endswith("/0"):
            node_id = f"C_{len(concept_nodes)}"
            concept_nodes.append(node_id)
            G.add_node(node_id, onto_id=idx.split("/")[-2], label=label, vector=vec)

    ontoid_to_nodeid = {G.nodes[n]["onto_id"]: n for n in G.nodes}

    # Add ontology structure edges
    if use_ontology_structure:
        with open(ontology_file_1, 'r') as f:
            onto_data = json.load(f)

        edges = []
        if "graphs" in onto_data:  # OBO-style
            for e in onto_data["graphs"][0]["edges"]:
                src, tgt = e["obj"].split("/")[-1], e["sub"].split("/")[-1]
                if src in ontoid_to_nodeid.keys() and tgt in ontoid_to_nodeid.keys():
                    edges.append((ontoid_to_nodeid[src], ontoid_to_nodeid[tgt], 1.0))
        else:  # Custom JSON style
            for e in onto_data:
                if len(e["Parents"]) == 0: continue
                tgt = e["Class ID"].split("/")[-1]
                if tgt not in ontoid_to_nodeid.keys(): continue
                for p in e["Parents"]:
                    src = p.split("/")[-1]
                    if src in ontoid_to_nodeid.keys():
                        edges.append((ontoid_to_nodeid[src], ontoid_to_nodeid[tgt], 1.0))
        if ontology_file_2 != "None":
            with open(ontology_file_2, 'r') as f:
                onto_data = json.load(f)
            if "graphs" in onto_data:  # OBO-style
                for e in onto_data["graphs"][0]["edges"]:
                    src, tgt = e["obj"].split("/")[-1], e["sub"].split("/")[-1]
                    if src in ontoid_to_nodeid.keys() and tgt in ontoid_to_nodeid.keys():
                        edges.append((ontoid_to_nodeid[src], ontoid_to_nodeid[tgt], 1.0))
            else:  # Custom JSON style
                for e in onto_data:
                    if len(e["Parents"]) == 0: continue
                    tgt = e["Class ID"].split("/")[-1]
                    if tgt not in ontoid_to_nodeid.keys(): continue
                    for p in e["Parents"]:
                        src = p.split("/")[-1]
                        if src in ontoid_to_nodeid.keys():
                            edges.append((ontoid_to_nodeid[src], ontoid_to_nodeid[tgt], 1.0))

        G.add_weighted_edges_from(edges)
        print(f"[Ontology] Added {len(edges)} edges.")

    # Add semantic similarity edges
    if use_semantic_similarity:
        ann = ApproximateNearestNeighborSearch()
        vectors = np.array([G.nodes[n]["vector"] for n in concept_nodes])
        ann.make_index(vectors, concept_nodes, [G.nodes[n]["label"] for n in concept_nodes])

        added_edges = 0
        for node in tqdm(concept_nodes, desc="Adding semantic similarity edges"):
            results, scores = ann.search(np.array([G.nodes[node]["vector"]]), L + 1)
            neighbors = [c for i, c in enumerate(results[0]) if c != node]
            selected, dists = acquire_neighbors(node, neighbors, M, G, dist_func=DistanceL2)
            for target, dist in zip(selected, dists):
                G.add_edge(node, target, weight=1 / (1 + dist))
                added_edges += 1

        print(f"[Semantic] Added {added_edges} edges.")

    print(f"Final graph: {len(G.nodes)} nodes, {len(G.edges)} edges.")
    with open(output_file, 'wb') as f:
        pickle.dump(G, f)

    return G


# ---------- Example usage ----------
if __name__ == "__main__":
    build_graph(
        concept_file="models/hoip/embedding/ontology_concept_embeddings.pkl",
        candidate_file="../data/onto/hoip/target_concept_id_list.json",
        ontology_file_1="../data/onto/hoip/go-basic.json",
        ontology_file_2="../data/onto/hoip/hoip_ontology.json",
        output_file="models/hoip/embedding/graph_ossi.pkl",
        use_semantic_similarity=True,
        use_ontology_structure=True,
        L=10,
        M=10
    )
