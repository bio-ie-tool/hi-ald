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
