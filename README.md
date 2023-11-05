# RepNet for Quantifying the Reproducibility of Graph Neural Networks
Please contact hizircanbayram@gmail.com for inquiries. Thanks. 

# Introduction
This work is accepted at the MICCAI PRIME workshop 2023.

>**RepNet for Quantifying the Reproducibility of Graph Neural Networks in Multiview Brain Connectivity Biomarker Discovery**
>
> Hızır Can Bayram, Mustafa Serdar Çelebi and Islem Rekik
>
> BASIRA Lab, Imperial-X and Department of Computing, Imperial College London, London, UK
> Istanbul Technical University, Istanbul, Turkey
>
> **Abstract:** *The brain is a highly complex multigraph that presents different types of connectivities (i.e., edge types) linking pairs of anatomical regions. Each edge type captures a connectional view of the neural roadmaps such as function and morphology. Recently, several graph neural networks (GNNs) have been used to learn the representation of the brain multigraph for disease classification, stratification and outcome prediction. However, such works primarily focus on boosting the accuracy performance without investigating the reproducibility of a trained GNN model in biomarker discovery across data views and perturbations in distribution. Here, we propose RepNet, a framework that ranks various GNN architectures by quantifying their biomarker reproducibility. Specifically, we lay the mathematical foundation for inter-view and inter-model reproducibility and validates it through extensive intra-and and inter-consistency experiments. Our results showed that RepNet can identify the most reproducible GNN model, able to identify trustworthy and consistent biomarkers with a superior performance to state-of-the-art methods.*

##  RepNet framework for quantifying the reproducibility of graph neural networks
![mainfigure](https://github.com/basiralab/RepNet/assets/23126077/849f834f-9060-4bca-bc03-066e924f498a)

## Code
This code was implemented using Python 3.9.13

## Requirements
torch = 1.12.1 + cpu \
numpy = 1.23.1 \
scipy = 1.9.2 \
torch_geometric = 2.1.0.post1 \
sklearn = 1.1.2 \
pandas = 1.5.0

## Components of RepNet Code
| Component | Content |
| ------ | ------ |
| demo.py | Starts the process. |
| Analysis.ipynb | Includes the generation of results, tables, and comparisons. |
| cross_val.py | Prepares and returns the DataLoaders. |
| extract_results.py | Extracts and normalizes the weights of the models. |
| main_diffpool.py | Performs all the training tasks for the DiffPool model. |
| main_gat.py | Performs all the training tasks for the GAT model. |
| main_gcn.py | Performs all the training tasks for the GCN model. |
| main_gunet.py | Performs all the training tasks for the Gunet model. |
| main_sag.py | Performs all the training tasks for the SAG model. |
| models_diffpool.py | The architecture of the DiffPool model. |
| models_gat.py | The architecture of the GAT model. |
| models_gcn.py | The architecture of the GCN model. |
| models_gunet.py | The architecture of the Gunet model. |
| models_sag.py | The architecture of the SAG model. |
| reorganizeDataset.py | Creates the 'edges' and 'label' files from the .mat files. |
| simulate_data.py | Generates simulation data in a processable format. |
| split_cv.py | Cross-validates the dataset based on the specified hyperparameters. |

## External Validation
![table4](https://github.com/basiralab/RepNet/assets/23126077/ef3a15aa-b24e-4c8f-9724-f3f21232d7d0)
We display the IQR scores calculated by index ranks of each biomarker obtained from different trainings of each GNN model. IQR scores shows the deviation of each biomarker across different trainings of each GNN. A few insights into the reproducibility of a GNN model can be inferred by examining the IQR scores, where biomarkers having IQR scores closer to 0 indicate better reproducibility. We observe that GCN has higher reproducibility than DiffPool, as reflected by the lower IQR scores across all biomarkers. We also notice that both GCN and DiffPool have lower IQR scores than the other three models, indicating greater reproducibility. Another significant finding is that there is a noticeable quantitative difference between the reproducibility of GCN and DiffPool compared to the other models. Our findings are more consistent with RepNet quantification method, providing further evidence of RepNet superiority over RG-Select.

## Paper Link
*https://link.springer.com/chapter/10.1007/978-3-031-46005-0_4*

## Citation
```latex
@inproceedings{bayram2023repnet,
  title={RepNet for Quantifying the Reproducibility of Graph Neural Networks in Multiview Brain Connectivity Biomarker Discovery},
  author={Bayram, Hizir Can and {\c{C}}elebi, Mehmet Serdar and Rekik, Islem},
  booktitle={International Workshop on PRedictive Intelligence In MEdicine},
  pages={35--45},
  year={2023},
  organization={Springer}
}
```
## License
Our code is released under MIT License (see LICENSE file for details).



