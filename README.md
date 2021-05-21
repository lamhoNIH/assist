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

<table>
    <thead>
        <tr>
            <th></th>
	    <th><sub>File</sub></th>
            <th><sub>Description</sub></th>
	    <th><sub>Human</sub></th>
	    <th><sub>Mouse</sub></th>
        </tr>
    </thead>
    <tbody>
        <tr>
		    <td><sub>Input</sub></td>
	        <td><sub>PFC_HDID_norm_exp.txt</sub></td>
	        <td><sub>normalized counts from RNA-seq or microarray</sub></td>
		    <td><sub</sub></td>
		    <td><sub>:heavy_check_mark:</sub></td>
        </tr>
        <tr>
	        <tr>
			    <td rowspan=2><sub>Output</sub></td>
		        <td><sub>tom.csv</sub></td>
		        <td><sub>TOM co-expression network</sub></td>
			    <td><sub></sub></td>
			    <td><sub>:heavy_check_mark:</sub></td>
	        </tr>
	        <tr>
		        <td><sub>wgcna_modules.csv</sub></td>
		        <td><sub>gene module assignment by WGCNA hierarchical clustering</sub></td>
			    <td><sub></sub></td>
			    <td><sub>:heavy_check_mark:</sub></td>
	        </tr>
        </tr>
    </tbody>
</table>

**2. Module Extraction**

<table>
    <thead>
        <tr>
	        <th></th>
		    <th><sub>File</sub></th>
	        <th><sub>Description</sub></th>
		    <th><sub>Human</sub></th>
		    <th><sub>Mouse</sub></th>
        </tr>
    </thead>
    <tbody>
        <tr>
	        <tr>
			    <td rowspan=2><sub>Input</sub></td>
		        <td><sub>Kapoor_TOM.csv</sub></td>
		        <td><sub>TOM co-expression network</sub></td>
			    <td><sub>:heavy_check_mark:</sub></td>
			    <td><sub></sub></td>
	        </tr>
            <tr>
		        <td><sub>tom.csv</sub></td>
		        <td><sub>TOM co-expression network</sub></td>
			    <td><sub></sub></td>
			    <td><sub>:heavy_check_mark:</sub></td>
	        </tr>
        </tr>
        <tr>
	        <tr>
			    <td rowspan=2><sub>Output</sub></td>
		        <td><sub>network_louvain_default.csv</sub></td>
		        <td><sub>gene module assignment by Louvain algorithm using its default setting</sub></td>
			    <td><sub>:heavy_check_mark:</sub></td>
			    <td><sub>:heavy_check_mark:</sub></td>
	        </tr>
	        <tr>
		        <td><sub>network_louvain_agg1.csv</sub></td>
		        <td><sub>gene module assignment by Louvain algorithm using a different setting</sub></td>
			    <td><sub>:heavy_check_mark:</sub></td>
			    <td><sub>:heavy_check_mark:</sub></td>
	        </tr>
        </tr>
</table>

**3. Module Membership Analysis**

<table>
    <thead>
        <tr>
	        <th></th>
		    <th><sub>File</sub></th>
	        <th><sub>Description</sub></th>
		    <th><sub>Human</sub></th>
		    <th><sub>Mouse</sub></th>
        </tr>
    </thead>
    <tbody>
        <tr>
	        <tr>
			    <td rowspan=4><sub>Input</sub></td>
		        <td><sub>kapoor_wgcna_modules.csv</sub></td>
		        <td><sub>gene module assignment by WGCNA hierarchical clustering</sub></td>
			    <td><sub>:heavy_check_mark:</sub></td>
			    <td><sub></sub></td>
	        </tr>
            <tr>
		        <td><sub>wgcna_modules.csv</sub></td>
		        <td><sub>gene module assignment by WGCNA hierarchical clustering</sub></td>
			    <td><sub></sub></td>
			    <td><sub>:heavy_check_mark:</sub></td>
	        </tr>
            <tr>
		        <td><sub>network_louvain_default.csv</sub></td>
		        <td><sub>gene module assignment by Louvain algorithm using its default setting</sub></td>
			    <td><sub>:heavy_check_mark:</sub></td>
			    <td><sub>:heavy_check_mark:</sub></td>
	        </tr>
            <tr>
		        <td><sub>network_louvain_agg1.csv</sub></td>
		        <td><sub>gene module assignment by Louvain algorithm using a different setting</sub></td>
			    <td><sub>:heavy_check_mark:</sub></td>
			    <td><sub>:heavy_check_mark:</sub></td>
	        </tr>
        </tr>
        <tr>
	        <tr>
			    <td rowspan=3><sub>Output</sub></td>
		        <td><sub>plot_gene_cnt_each_cluster_wgcna.png</sub></td>
		        <td><sub>number of gene per module for WGCNA module assignment</sub></td>
			    <td><sub>:heavy_check_mark:</sub></td>
			    <td><sub>:heavy_check_mark:</sub></td>
	        </tr>
	        <tr>
		        <td><sub>plot_gene_cnt_each_cluster_louvain 1.png</sub></td>
		        <td><sub>number of gene per module for Louvain module assignment # 1</sub></td>
			    <td><sub>:heavy_check_mark:</sub></td>
			    <td><sub>:heavy_check_mark:</sub></td>
	        </tr>
	        <tr>
		        <td><sub>plot_gene_cnt_each_cluster_louvain 2.png</sub></td>
		        <td><sub>number of gene per module for Louvain module assignment # 2</sub></td>
			    <td><sub>:heavy_check_mark:</sub></td>
			    <td><sub>:heavy_check_mark:</sub></td>
	        </tr>
        </tr>
    </tbody>
</table>

**4. Module DE/Diagnostic Correlation**

<table>
    <thead>
        <tr>
	        <th></th>
		    <th><sub>File</sub></th>
	        <th><sub>Description</sub></th>
		    <th><sub>Human</sub></th>
		    <th><sub>Mouse</sub></th>
        </tr>
    </thead>
    <tbody>
        <tr>
	        <tr>
			    <td rowspan=7><sub>Input</sub></td>
		        <td><sub>deseq.alc.vs.control.age.rin.batch.gender.PMI. corrected.w.prot.coding.gene.name.xlsx</sub></td>
		        <td><sub>differential expression analysis</sub></td>
			    <td><sub>:heavy_check_mark:</sub></td>
			    <td><sub></sub></td>
	        </tr>
	        <tr>
		        <td><sub>de_data.csv</sub></td>
		        <td><sub>differential expression analysis</sub></td>
			    <td><sub></sub></td>
			    <td><sub>:heavy_check_mark:</sub></td>
	        </tr>
	        <tr>
		        <td><sub>kapoor_expression_Apr5.txt</sub></td>
		        <td><sub>normalized counts from RNA-seq or microarray</sub></td>
			    <td><sub>:heavy_check_mark:</sub></td>
			    <td><sub></sub></td>
	        </tr>
            <tr>
		        <td><sub>kapoor_wgcna_modules.csv</sub></td>
		        <td><sub>gene module assignment by WGCNA hierarchical clustering</sub></td>
			    <td><sub>:heavy_check_mark:</sub></td>
			    <td><sub></sub></td>
	        </tr>
            <tr>
		        <td><sub>wgcna_modules.csv</sub></td>
		        <td><sub>gene module assignment by WGCNA hierarchical clustering</sub></td>
			    <td><sub></sub></td>
			    <td><sub>:heavy_check_mark:</sub></td>
	        </tr>
            <tr>
		        <td><sub>network_louvain_default.csv</sub></td>
		        <td><sub>gene module assignment by Louvain algorithm using its default setting</sub></td>
			    <td><sub>:heavy_check_mark:</sub></td>
			    <td><sub>:heavy_check_mark:</sub></td>
	        </tr>
            <tr>
		        <td><sub>network_louvain_agg1.csv</sub></td>
		        <td><sub>gene module assignment by Louvain algorithm using a different setting</sub></td>
			    <td><sub>:heavy_check_mark:</sub></td>
			    <td><sub>:heavy_check_mark:</sub></td>
	        </tr>
        </tr>
        <tr>
	        <tr>
			    <td rowspan=4><sub>Output</sub></td>
		        <td><sub>expression_meta.csv</sub></td>
		        <td><sub>normalized expression data joined with subjects' metadata</sub></td>
			    <td><sub>:heavy_check_mark:</sub></td>
			    <td><sub></sub></td>
	        </tr>
	        <tr>
		        <td><sub>cluster_DE_perc_xx.png</sub></td>
		        <td><sub>DEG distribution across modules</sub></td>
			    <td><sub>:heavy_check_mark:</sub></td>
			    <td><sub>:heavy_check_mark:</sub></td>
	        </tr>
	        <tr>
		        <td><sub>plot_sig_perc_xx.png</sub></td>
		        <td><sub>% genes in the module that are significant for different alcohol trait group</sub></td>
			    <td><sub>:heavy_check_mark:</sub></td>
			    <td><sub>:heavy_check_mark:</sub></td>
	        </tr>
	        <tr>
		        <td><sub>cluster_phenotype_corr_xx.png</sub></td>
		        <td><sub>module eigengene and alcohol trait correlation</sub></td>
			    <td><sub>:heavy_check_mark:</sub></td>
			    <td><sub>:heavy_check_mark:</sub></td>
	        </tr>
        </tr>
    </tbody>
</table>

**5. Module Network Embedding**

<table>
    <thead>
        <tr>
	        <th></th>
		    <th><sub>File</sub></th>
	        <th><sub>Description</sub></th>
		    <th><sub>Human</sub></th>
		    <th><sub>Mouse</sub></th>
        </tr>
    </thead>
    <tbody>
        <tr>
	        <tr>
			    <td rowspan=7><sub>Input</sub></td>
		        <td><sub>Kapoor_TOM.csv</sub></td>
		        <td><sub>TOM co-expression network</sub></td>
			    <td><sub>:heavy_check_mark:</sub></td>
			    <td><sub></sub></td>
	        </tr>
	        <tr>
		        <td><sub>tom.csv</sub></td>
		        <td><sub>TOM co-expression network</sub></td>
			    <td><sub></sub></td>
			    <td><sub>:heavy_check_mark:</sub></td>
	        </tr>
	        <tr>
		        <td><sub>deseq.alc.vs.control.age.rin.batch.gender.PMI. corrected.w.prot.coding.gene.name.xlsx</sub></td>
		        <td><sub>differential expression analysis</sub></td>
			    <td><sub>:heavy_check_mark:</sub></td>
			    <td><sub></sub></td>
	        </tr>
            <tr>
		        <td><sub>de_data.csv</sub></td>
		        <td><sub>differential expression analysis</sub></td>
			    <td><sub></sub></td>
			    <td><sub>:heavy_check_mark:</sub></td>
	        </tr>
            <tr>
		        <td><sub>network_louvain_default.csv</sub></td>
		        <td><sub>gene module assignment chosen to compare with embedding clusters (user's choice)</sub></td>
			    <td><sub>:heavy_check_mark:</sub></td>
			    <td><sub></sub></td>
	        </tr>
            <tr>
		        <td><sub>wgcna_modules.csv</sub></td>
		        <td><sub>gene module assignment chosen to compare with embedding clusters (user's choice)</sub></td>
			    <td><sub></sub></td>
			    <td><sub>:heavy_check_mark:</sub></td>
	        </tr>
            <tr>
		        <td><sub>expression_meta.csv</sub></td>
		        <td><sub>normalized expression data joined with subjects' metadata</sub></td>
			    <td><sub>:heavy_check_mark:</sub></td>
			    <td><sub></sub></td>
	        </tr>
        </tr>
        <tr>
	        <tr>
			    <td rowspan=9><sub>Output</sub></td>
		        <td><sub>embedding.csv</sub></td>
		        <td><sub>network embedding</sub></td>
			    <td><sub>:heavy_check_mark:</sub></td>
			    <td><sub>:heavy_check_mark:</sub></td>
	        </tr>
	        <tr>
		        <td><sub>plot_gene_cnt_each_cluster_Network.png</sub></td>
		        <td><sub>number of gene per network module (same as the output in ```Module Membership Analysis```</sub></td>
			    <td><sub>:heavy_check_mark:</sub></td>
			    <td><sub>:heavy_check_mark:</sub></td>
	        </tr>
	        <tr>
		        <td><sub>plot_gene_cnt_each_cluster_epoch=5_alpha=0.1.png</sub></td>
		        <td><sub>number of gene per cluster for WGCNA module assignment (compare it with plot_gene_cnt_each_cluster_Network.png)</sub></td>
			    <td><sub>:heavy_check_mark:</sub></td>
			    <td><sub>:heavy_check_mark:</sub></td>
	        </tr>
	        <tr>
		        <td><sub>cluster_DE_perc_network.png</sub></td>
		        <td><sub>DEG distribution across modules (same as the output in ```Module DE/Diagnostic Correlation```</sub></td>
			    <td><sub>:heavy_check_mark:</sub></td>
			    <td><sub>:heavy_check_mark:</sub></td>
	        </tr>
	        <tr>
		        <td><sub>cluster_DE_perc_epoch=5_alpha=0.1 embedding.png</sub></td>
		        <td><sub>DEG distribution across embedding clusters (compare it with cluster_DE_perc_network.png)</sub></td>
			    <td><sub>:heavy_check_mark:</sub></td>
			    <td><sub>:heavy_check_mark:</sub></td>
	        </tr>
	        <tr>
		        <td><sub>cluster_phenotype_corr_network.png</sub></td>
		        <td><sub>module eigengene and alcohol trait correlation (same as the output in ```Module DE/Diagnostic Correlation```</sub></td>
			    <td><sub>:heavy_check_mark:</sub></td>
			    <td><sub>:heavy_check_mark:</sub></td>
	        </tr>
	        <tr>
		        <td><sub>cluster_phenotype_corr_embedding.png</sub></td>
		        <td><sub>cluster eigengene and alcohol trait correlation (compare it with cluster_phenotype_corr_network.png)</sub></td>
			    <td><sub>:heavy_check_mark:</sub></td>
			    <td><sub>:heavy_check_mark:</sub></td>
	        </tr>
	        <tr>
		        <td><sub>alcohol trait correlation network vs embedding.png</sub></td>
		        <td><sub>distribution plot to compare cluster_phenotype_corr_network.png and cluster_phenotype_corr_embedding.png</sub></td>
			    <td><sub>:heavy_check_mark:</sub></td>
			    <td><sub>:heavy_check_mark:</sub></td>
	        </tr>
	        <tr>
		        <td><sub>cluster_jaccard_Network vs epoch=5_alpha=0.1.png</sub></td>
		        <td><sub>pairwise jaccard comparison to determine network module and embedding cluster similarity</sub></td>
			    <td><sub>:heavy_check_mark:</sub></td>
			    <td><sub>:heavy_check_mark:</sub></td>
	        </tr>
        </tr>
    </tbody>
</table>

**6. ML and Critical Gene Identifier**

<table>
    <thead>
        <tr>
	        <th></th>
		    <th><sub>File</sub></th>
	        <th><sub>Description</sub></th>
		    <th><sub>Human</sub></th>
		    <th><sub>Mouse</sub></th>
        </tr>
    </thead>
    <tbody>
        <tr>
	        <tr>
			    <td rowspan=5><sub>Input</sub></td>
		        <td><sub>Kapoor_TOM.csv</sub></td>
		        <td><sub>TOM co-expression network</sub></td>
			    <td><sub>:heavy_check_mark:</sub></td>
			    <td><sub></sub></td>
	        </tr>
	        <tr>
		        <td><sub>embedding.csv</sub></td>
		        <td><sub>network embedding</sub></td>
			    <td><sub>:heavy_check_mark:</sub></td>
			    <td><sub>:heavy_check_mark:</sub></td>
	        </tr>
	        <tr>
		        <td><sub>expression_meta.csv</sub></td>
		        <td><sub>normalized expression data joined with subjects' metadata</sub></td>
			    <td><sub>:heavy_check_mark:</sub></td>
			    <td><sub></sub></td>
	        </tr>
            <tr>
		        <td><sub>deseq.alc.vs.control.age.rin.batch.gender.PMI. corrected.w.prot.coding.gene.name.xlsx</sub></td>
		        <td><sub>differential expression analysis</sub></td>
			    <td><sub>:heavy_check_mark:</sub></td>
			    <td><sub></sub></td>
	        </tr>
            <tr>
		        <td><sub>de_data.csv</sub></td>
		        <td><sub>differential expression analysis</sub></td>
			    <td><sub></sub></td>
			    <td><sub>:heavy_check_mark:</sub></td>
	        </tr>
        </tr>
        <tr>
	        <tr>
			    <td rowspan=9><sub>Output</sub></td>
		        <td><sub>critical_genes.csv</sub></td>
		        <td><sub>candidate genes identified by ASSIST</sub></td>
			    <td><sub>:heavy_check_mark:</sub></td>
			    <td><sub>:heavy_check_mark:</sub></td>
	        </tr>
	        <tr>
		        <td><sub>neighbor_genes.csv</sub></td>
		        <td><sub>closest DEG neighbors in the co-expression network</sub></td>
			    <td><sub>:heavy_check_mark:</sub></td>
			    <td><sub>:heavy_check_mark:</sub></td>
	        </tr>
	        <tr>
		        <td><sub>run_ml_.pngg</sub></td>
		        <td><sub>machine learning accuracy</sub></td>
			    <td><sub>:heavy_check_mark:</sub></td>
			    <td><sub>:heavy_check_mark:</sub></td>
	        </tr>
	        <tr>
		        <td><sub>run_ml_top_dims.png</sub></td>
		        <td><sub>machine learning accuracy using only the most important dimensions</sub></td>
			    <td><sub>:heavy_check_mark:</sub></td>
			    <td><sub>:heavy_check_mark:</sub></td>
	        </tr>
	        <tr>
		        <td><sub>gene_phenotype_corr_for_xx.png</sub></td>
		        <td><sub>critical gene/DEG/neighbor gene correlation with alcohol traits</sub></td>
			    <td><sub>:heavy_check_mark:</sub></td>
			    <td><sub></sub></td>
	        </tr>
	        <tr>
		        <td><sub>alcohol trait correlation CG, neighbor & DEG.png</sub></td>
		        <td><sub>distribution plot to compare gene_phenotype_corr_for_xx.png</sub></td>
			    <td><sub>:heavy_check_mark:</sub></td>
			    <td><sub></sub></td>
	        </tr>
	        <tr>
		        <td><sub>jaccard_average_Important dim overlap within model repeats.png</sub></td>
		        <td><sub>the important dimensions overlap between the repeats of each model</sub></td>
			    <td><sub>:heavy_check_mark:</sub></td>
			    <td><sub>:heavy_check_mark:</sub></td>
	        </tr>
	        <tr>
		        <td><sub>jaccard_critical_genes_Critical gene overlap between models.png</sub></td>
		        <td><sub>critical gene overlap between each two models</sub></td>
			    <td><sub>:heavy_check_mark:</sub></td>
			    <td><sub>:heavy_check_mark:</sub></td>
	        </tr>
	        <tr>
		        <td><sub>plot_nearby_impact_num_.png</sub></td>
		        <td><sub>top 10 critical genes</sub></td>
			    <td><sub>:heavy_check_mark:</sub></td>
			    <td><sub>:heavy_check_mark:</sub></td>
	        </tr>
        </tr>
    </tbody>
</table>


