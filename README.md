# Scaling Biomedical Concept Recognition with LLM-based Auto-Labeling and Hierarchical Indexing

We will update the repository soon.

This repository includes:
   - The constructed auto-labelled datasets for the HPA concept and the HoIP concept.
   - The processing scripts for LLM-generated outputs from each module in our auto-labelling pipeline.
   - The scripts for index construction.


## Auto-labelled datasets: HPA ALD and HoIP ALD
You can find the compressed files for our ALDs in the "data/ald" directory.

Basic statistics:

| Dataset  | Passage | Concept | Unique Concept | Q=Average | Q=Good | Q=Excellent |
|----------| ------- | ------- | -------------- |-----------| ------ | ----------- |
| HPA ALD  | 54301 | 197824 | 12725 | 25511     | 28758 | 32 |
| HoIP ALD | 34097 | 370461 | 15976 | 17781     | 16306 | 10 |

The construction includes 5 stages, as shown in Figure 1. 
Only instances rated as "average/good/excellent" are included in our ALDs.

Figure 1: Overview of the Auto-labeling Pipeline. Given an input passage, the pipeline begins by generating  intermediate claims (Arrow 1), followed by candidate concept names derived from these claims (Arrow 2). The  resulting names are semantically matched to ontology terms via a k-Nearest Neighbor (kNN) search, forming a  preliminary list of concept candidates (PCC), shown in the blue-shaded table. Following the PCC stage, concept  classification (CC), relabeling (RL), and guideline-based filtering (GF) steps are applied to get the final annotations (highlighted in bold). The instance will be taken as a training instance if its quality meets the requirement (QS).

![ald-pipeline-simple](https://github.com/user-attachments/assets/e944b65f-2a77-41a3-b24a-20eae5032c44)

The ID of each concept is simplified, for example, "http://purl.obolibrary.org/obo/GO_0006954" is simplified as "GO_0006954". You can add the prefix "http://purl.obolibrary.org/obo/" before each concept ID to get the official page of each ontology concept.


## Index construction - OSSI

There are 3 steps for constructing the Ontology-Semantic Search Index (OSSI).

Step 1: Get semantic vector embeddings of all concept names and synonyms.

Step 2: Build a graph given the ontology and semantic information.

Step 3: Do hierarchical graph partitioning by the Louvain algorithm.

Step 4: Map the new index to the ontology concept.

### Step 1: Get semantic vector embeddings
You can run the ``scripts/get_concept_embeddings.py`` to get semantic embeddings of target concept names/synonyms.
Specifically, we get names and synonyms from a given ontology JSON file and a list of target concept IDs, vectorize each name/synonym using a Transformer model (e.g., SapBERT), and store the result as a pickle.

Before you run the script, assign the file paths and the card of the embedding model in the script well, such as: 

```python
ontology_file = "../data/onto/hoip/concept_info.json"
target_ids_file = "../data/onto/hoip/target_concept_id_list.json"
embedding_model = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
output_file = "models/hoip/embedding/ontology_concept_embeddings.pkl"
```

We provide ``concept_info.json`` and ``target_concept_id_list.json`` used in our experiments.

In ``concept_info.json``, the information of each concept we got from the ontology is organized in ``"concept_id": ["name", "definition", "a list of tree paths", "a list of synonyms"]`` format. As the file included "tree path" is too large to share, we set the corresponding place as ``[]``.
For HoIP concepts, the ``concept_info.json`` includes all biological process concepts in the HoIP Ontology.
For HPA concepts, the ``concept_info.json`` includes all concepts in the HPO.

``target_concept_id_list.json`` is a list of target concepts that we want to identify. 

For HoIP concepts, the ``target_concept_id_list.json`` includes all biological process concepts presented in both the HoIP Ontology and the Gene Ontology.
For HPA concepts, the ``target_concept_id_list.json`` includes all phenotype abnormality concepts in the HPO.

Basic statistics of target concepts:

| Concept Type | Concept | Name    | Synonym  | 
|--------------|---------|---------|----------|
| HPA          | 18354   | 18354   | 21987    |
| HoIP         | 29367   | 29367   | 87705    | 


### Step 2: Build a graph

A graph is built from the concepts in the ontology, where each concept is a node in the graph and the edges come from the ontology structure (OSI), semantic similarity (SSI), or a combination of both (OSSI).

Set the file paths well before you run the ``scripts/build_graph.py``, such as:

```python
concept_file="models/hoip/embedding/ontology_concept_embeddings.pkl",
candidate_file="../data/onto/hoip/target_concept_id_list.json",
ontology_file_1="../data/onto/hoip/go_basic.json",
ontology_file_2="../data/onto/hoip/hoip_ontology.json",
output_file="models/hoip/embedding/graph_ossi.pkl",
use_semantic_similarity=True,
use_ontology_structure=True,
L=10,
M=10
```

For HoIP concepts, there are two sources of ontology structure information: [GO-basic.json](https://purl.obolibrary.org/obo/go/go-basic.json) from Gene Ontology, [hoip_ontology.json](https://github.com/norikinishida/hoip-dataset/tree/main/releases) from HoIP ontology.
We recommend using both of them. You can download them and place them into ``../data/onto/hoip`` directory, so that the setting is

```python
ontology_file_1="../data/onto/hoip/go-basic.json",
ontology_file_2="../data/onto/hoip/hoip_ontology.json",
```

For the HPA concept, only the information from HPO is used, so that the setting is 
```python
ontology_file_1="../data/onto/hoip/hpa_ontology.json",
ontology_file_2="None",
```

### Step 3: Do hierarchical graph partition
After constructing the ontology-based graph from embeddings, this step recursively partitions the graph using the Louvain community detection algorithm with dynamic resolution. The result is a multi-level hierarchical clustering of ontology concepts.

Please set the file paths well before you run ``scripts/partition_graph.py``, such as:

```python
args.input_file = "models/hoip/embedding/graph_ossi.pkl"
args.output_file = "models/hoip/embedding/graph_ossi_louvain.pkl"
```

If there are some internal nodes containing too many children (> 10 nodes), the script will output a list of such internal nodes, like:
```text
Too big community: []
Too small community: ['0-0-6-2-7-0-4-5-3-0', '0-0-6-2-7-0-4-5-3-1', ...]
```

Please run ``scripts/partition_graph_metis.py`` and partition them using pymetis.

The file paths should be the same as those used in the Louvain partition:
```python
args.input_file = "../models/hoip/embedding/graph_ossi.pkl"
args.output_file = "../models/hoip/embedding/graph_ossi_louvain.pkl"
args.num_clusters = 10
args.resume = True
```

Note: ``resume=True`` is a necessary condition because the partitioning is based on Louvain's results to continue refining.

Process and Queue in multiprocessing are used to avoid memory problems caused by PyMetis in the main process. However, it is common that the script will get stuck and no response at all while it running. We tried hard to find the cause, but we failed. So you can stop the process and rerun the scripts everytime it there is no response, until your partition is *Finished*.


### Step 4: Map new index to ontology concept

The last step is to get the "search index-ontology concept" mapping files for the training/prediction of MA-COIR.

Set the following parameters well and run the script ``scripts/map_index.py``, such as:
```python
index_types = ["osi", "ssi", "ossi"]  # Variants of graph types or feature sets
concept_type = "hpa"  # Can be "hpa", "hoip", etc.
```

We provide the mapping files used in our experiments, all in ``data/search_index``.
