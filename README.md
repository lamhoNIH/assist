# ASSIST (Alcoholism Solutions: Synthesizing Information to Support Treatments)

This repo contains the source code of ASSIST analysis modules developed under the Phase I of the ASSIST project. These modules can be run in Jupyter notebooks, Docker containers, as well as workflows managed by the ADE (Active Discovery Engine) with increasing support for provenance.

### Description of the analysis modules:
|#| Analysis Module | Description |
|-|-------|-------------|
|1| Network Analysis | Network construction by WGCNA |
|2| Module Extraction | Network module detection by Louvain Algorithm |
|3| Module Membership Analysis | Genes count per module and module assignment stability check |
|4| Module DE/Diagnostic Correlation | Biological relevance of network module check |
|5| Module Network Embedding | Conversion of network to machine learning-friendly matrix representation |
|6| ML and Critical Gene Identifier | Machine learning to extract features for critical gene identification |

The analysis workflow follows the order of the modules and the modules are inter-related as depicted in the conceptual workflow below.
<p align="center">
  <img src="https://user-images.githubusercontent.com/12038408/117026434-ca74fa80-acc9-11eb-937c-ffaa7547ff34.png" width="700" height="650">
</p>

Module ```Critical Gene Validation``` requires a 3rd party license and is thus not included in this repo.

## User Guide
Below we describe how to set up and run the ASSIST analysis modules in three different modes.

### 1. How to set up the environment for Jupyter notebooks
Jupyter notebooks for ASSIST analysis modules are included to allow researchers to test out the analysis code using the Jupyter notebook interface.

### 2. How to launch containers for each analysis module

### 3. How to run ASSIST modules in a workflow using ADE

#### Prepare ADE runtime environment

#### Create a workflow module in ADE

#### Integrate workflow modules in ADE

#### How to navigate in ADE


For each of the analysis modules below, include detailed description on input data (file name, file content, columns the module cares about), what the analysis does about the input data, and what the output the module generates. This can include snapshots of sample dataframes and plots.

## For all the input/output data below, ```H``` means it's for the human example data. ```M``` means it's for the mouse example data.

**1. Network Analysis**

Note that for the Kapoor data used in our analysis (aka the human data), the ```Network Analysis``` module was skipped as the TOM network and the WGCNA module assignment were already published so the example for this module below is for the HDID mouse data. 
| Input | Description |
|-------|-------------|
| PFC_HDID_norm_exp.txt ```M``` | normalized counts from RNA-seq or microarray |

| Output | Description |
|-------|-------------|
| tom.csv ```M``` | TOM co-expression network |
| wgcna_modules.csv ```M``` | gene module assignment by WGCNA hierarchical clustering |

**2. Module Extraction**
| Input | Description |
|-------|-------------|
| Kapoor_TOM.csv ```H``` or tom.csv ```M``` | TOM co-expression network |

| Output | Description |
|-------|-------------|
| network_louvain_default.csv ```H``` ```M``` | gene module assignment by Louvain algorithm using its default setting |
| network_louvain_agg1.csv ```H``` ```M``` | gene module assignment by Louvain algorithm using a different setting |

**3. Module Membership Analysis**
| Input | Description |
|-------|-------------|
| kapoor_wgcna_modules.csv ```H``` or wgcna_modules.csv ```M``` | gene module assignment by WGCNA hierarchical clustering |
| network_louvain_default.csv ```H``` ```M``` | gene module assignment by Louvain algorithm using its default setting |
| network_louvain_agg1.csv ```H``` ```M``` | gene module assignment by Louvain algorithm using a different setting |

| Output | Description |
|-------|-------------|
| plot_gene_cnt_each_cluster_wgcna.png ```H``` ```M``` | number of gene per module for WGCNA module assignment |
| plot_gene_cnt_each_cluster_louvain 1.png ```H``` ```M``` | number of gene per module for Louvain module assignment # 1 |
| plot_gene_cnt_each_cluster_louvain 2.png ```H``` ```M``` | number of gene per module for Louvain module assignment # 2|

**4. Module DE/Diagnostic Correlation**
| Input | Description |
|-------|-------------|
| deseq.alc.vs.control.age.rin.batch.gender.PMI. corrected.w.prot.coding.gene.name.xlsx ```H``` or de_data.csv ```M``` | differential expression analysis |
| kapoor_expression_Apr5.txt ```H``` |  normalized counts from RNA-seq or microarray |
| kapoor_wgcna_modules.csv ```H``` or wgcna_modules.csv ```M``` | gene module assignment by WGCNA hierarchical clustering |
| network_louvain_default.csv ```H``` ```M``` | gene module assignment by Louvain algorithm using its default setting |
| network_louvain_agg1.csv ```H``` ```M``` | gene module assignment by Louvain algorithm using a different setting |

| Output | Description |
|-------|-------------|
| expression_meta.csv ```H``` | normalized expression data joined with subjects' metadata |
| cluster_DE_perc_xx.png ```H``` ```M``` | DEG distribution across modules |
| plot_sig_perc_xx.png ```H``` ```M``` | % genes in the module that are significant for different alcohol trait group |
| cluster_phenotype_corr_xx.png ```H``` ```M``` | module eigengene and alcohol trait correlation |

**5. Module Network Embedding**
| Input | Description |
|-------|-------------|
| Kapoor_TOM.csv ```H``` or tom.csv ```M``` | TOM co-expression network |
| deseq.alc.vs.control.age.rin.batch.gender.PMI. corrected.w.prot.coding.gene.name.xlsx ```H``` or de_data.csv ```M``` | differential expression analysis |
| network_louvain_default.csv ```H``` or wgcna_modules.csv ```M``` | gene module assignment chosen to compare with embedding clusters (user's choice) |
| expression_meta.csv ```H``` | normalized expression data joined with subjects' metadata |

| Output | Description |
|-------|-------------|
| embedding.csv ```H``` ```M``` | network embedding |
| plot_gene_cnt_each_cluster_Network.png ```H``` ```M``` | number of gene per network module (same as the output in ```Module Membership Analysis```) |
| plot_gene_cnt_each_cluster_epoch=5_alpha=0.1.png ```H``` ```M``` | number of gene per cluster for WGCNA module assignment (compare it with plot_gene_cnt_each_cluster_Network.png) |
| cluster_DE_perc_network.png ```H``` ```M``` | DEG distribution across modules (same as the output in ```Module DE/Diagnostic Correlation```) |
| cluster_DE_perc_epoch=5_alpha=0.1 embedding.png ```H``` ```M``` | DEG distribution across embedding clusters (compare it with cluster_DE_perc_network.png) |
| cluster_phenotype_corr_network.png ```H``` ```M``` |  module eigengene and alcohol trait correlation (same as the output in ```Module DE/Diagnostic Correlation```) |
| cluster_phenotype_corr_embedding.png ```H``` ```M``` | cluster eigengene and alcohol trait correlation (compare it with cluster_phenotype_corr_network.png) |
| alcohol trait correlation network vs embedding.png ```H``` ```M``` | distribution plot to compare cluster_phenotype_corr_network.png and cluster_phenotype_corr_embedding.png |
| cluster_jaccard_Network vs epoch=5_alpha=0.1.png ```H``` ```M``` | pairwise jaccard comparison to determine network module and embedding cluster similarity |

**6. ML and Critical Gene Identifier**
| Input | Description |
|-------|-------------|
| Kapoor_TOM.csv ```H``` | TOM co-expression network |
| embedding.csv ```H``` ```M``` | network embedding |
| expression_meta.csv ```H``` | normalized expression data joined with subjects' metadata |
| deseq.alc.vs.control.age.rin.batch.gender.PMI. corrected.w.prot.coding.gene.name.xlsx ```H``` or de_data.csv ```M``` | differential expression analysis |

| Output | Description |
|-------|-------------|
| critical_genes.csv ```H``` ```M``` | candidate genes identified by ASSIST |
| neighbor_genes.csv ```H``` ```M``` | closest DEG neighbors in the co-expression network |
| run_ml_.png ```H``` ```M``` | machine learning accuracy |
| run_ml_top_dims.png ```H``` ```M``` | machine learning accuracy using only the most important dimensions |
| gene_phenotype_corr_for_xx.png ```H``` | critical gene/DEG/neighbor gene correlation with alcohol traits |
| alcohol trait correlation CG, neighbor & DEG.png ```H``` | distribution plot to compare gene_phenotype_corr_for_xx.png |
| jaccard_average_Important dim overlap within model repeats.png ```H``` ```M``` | the important dimensions overlap between the repeats of each model |
| jaccard_critical_genes_Critical gene overlap between models.png ```H``` ```M``` | critical gene overlap between each two models |
| plot_nearby_impact_num_.png ```H``` ```M``` | top 10 critical genes |

