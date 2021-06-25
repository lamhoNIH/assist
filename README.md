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
<p align="center"><img src="https://user-images.githubusercontent.com/12038408/123433360-60762480-d599-11eb-911e-1af52a23df4d.png" width="700" height="650">
</p>


Module ```Critical Gene Validation``` requires a 3rd party license and is thus not included in this repo.

### Where to get data:
Download data from [ASSIST Dropbox](https://www.dropbox.com/sh/uajkuclelr409e3/AAC0hwI47Ssz8I_FeOH8Pplca/data?dl=0&subfolder_nav_tracking=1) in the `data` folder. The user guide below assumes that you have downloaded all data files and placed them under the `data` subfolder of this project.

## User Guide

Below we describe how to set up and run the ASSIST analysis modules in three different modes.

### 1. How to set up the environment for Jupyter notebooks
Jupyter notebooks for ASSIST analysis modules are included to allow researchers to test out the analysis code using the Jupyter notebook interface. The `notebooks` folder contains requirements files capturing software dependencies for the three notebooks included. Corresponding requirement file is loaded into each notebook at the beginning of the notebook.

### 2. How to launch containers for each analysis module
Before analysis modules can be launched through standalone containers, the corresponding images need to be loaded. You can either use the included Makefile to generate the corresponding images, or download them from [ASSIST Dropbox](https://www.dropbox.com/sh/uajkuclelr409e3/AAC0hwI47Ssz8I_FeOH8Pplca/data?dl=0&subfolder_nav_tracking=1) in the `images/standalone` folder, place them under the `images/standalone` subfolder of this project, and load them using:
```
make load-standalone-images
```
The analysis modules are meant to be launched in sequence in the order listed in the above table and there are configuration files in the `config` folder specifying all input files needed to launch the module and where the module will be generating its output files and plots. Before choosing an analysis module to execute, make sure all the input data specified in the corresponding config file are available.

There is a script called launch.py under the scripts folder that can be used to launch these analysis modules, e.g., to launch `Module Extraction` on the human dataset, use: `python launch.py module_extraction human <path to the data folder>`, where `<path to the data folder>` is the absolute path to the `data` folder under the project root.

### 3. How to run ASSIST modules in a workflow using ADE

#### Prepare ADE runtime environment
Download `ade_runtime.tgz` from [ASSIST Dropbox](https://www.dropbox.com/sh/uajkuclelr409e3/AAC0hwI47Ssz8I_FeOH8Pplca/data?dl=0&subfolder_nav_tracking=1) into the project root folder and unpack it using:
```
tar zxvf ade_runtime.tgz
```
This command will create the following folder structure under the project:
```
ade
├── bin
│   ├── launcher.bat
│   └── launcher.sh
├── create_node_docker_image.py
├── doc
│   ├── README.md
│   ├── action_props.gif
│   ├── connect.gif
│   ├── disconnect.gif
│   ├── doc_props.gif
│   ├── docker_props.png
│   ├── dynamic_props.gif
│   ├── export_data.gif
│   ├── launch.gif
│   ├── new_node.gif
│   ├── persist.gif
│   ├── remove_node.gif
│   ├── scroll_props.gif
│   ├── view_data.gif
│   ├── view_props.gif
│   └── workflow.png
└── repo
    ├── FastInfoset-1.2.16.jar
    ├── ST4-4.0.8.jar
    ├── ade-backend-1.0.0-SNAPSHOT.jar
    ├── ade-frontend-1.0.0-SNAPSHOT.jar
    ├── ade-launcher-1.0.0-SNAPSHOT.jar
...
```


#### Use ADE to run analysis workflow
Use the launch script (`launch.bat` or `launch.sh`) to start up the ADE workflow user interface. There are ready made workflows for both `human` and `mouse` datasets under the `workflows` folder of this repo that can be loaded into the user interface. For this, you need to first download the ADE images for the analysis modules from [ASSIST Dropbox](https://www.dropbox.com/sh/uajkuclelr409e3/AAC0hwI47Ssz8I_FeOH8Pplca/data?dl=0&subfolder_nav_tracking=1) in the `images/ade` folder and place them under `~/.ade_image_repo/netrias` on your machine. This is where ADE will be loading images into the runtime environment.

Follow [ADE documentation](./ade/doc/README.md) that provides detailed description on using the ADE user interface.


#### Detailed description of analysis modules

## For all the input/output data below, `Human` means it's for the human example data (Kapoor et al 2019). `Mouse` means it's for the mouse example data (Ferguson et al 2019).
The difference is because the two example datasets (Kapoor and HDID) we used had difference in the availability of the data. For example, `TOM co-expression network` and `gene module assignment by WGCNA hierarchical clustering` for the human data were provided to us but not available for the mouse data so the `Network Analysis` had to be run to construct these two files for the mouse. `subjects' alcohol metadata` was only available for the human data so all the analyses that involve diagnostics were skipped for the mouse data. 

**1. Network Analysis**

Note that for the Kapoor data used in our analysis (aka the human data), the `Network Analysis` module was skipped as the TOM network and the WGCNA module assignment were already published so the example for this module below is for the HDID mouse data. 

<table border="1">
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
    </tbody>
    <tbody>
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
    </tbody>
</table>

**2. Module Extraction**

<table border="1">
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
    </tbody>
    <tbody>
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
    </tbody>
</table>

**3. Module Membership Analysis**

<table border="1">
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
    </tbody>
    <tbody>
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
    </tbody>
</table>

**4. Module DE/Diagnostic Correlation**

<table border="1">
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
        <td rowspan=8><sub>Input</sub></td>
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
        <td><sub>Kapoor2019_coga.inia.detailed.pheno.04.12.17.csv</sub></td>
        <td><sub>subjects' alcohol metadata</sub></td>
        <td><sub>:heavy_check_mark:</sub></td>
        <td><sub></sub></td>
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
    </tbody>
    <tbody>
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
        <td><sub></sub></td>
      </tr>
      <tr>
        <td><sub>cluster_phenotype_corr_xx.png</sub></td>
        <td><sub>module eigengene and alcohol trait correlation</sub></td>
        <td><sub>:heavy_check_mark:</sub></td>
        <td><sub></sub></td>
      </tr>
    </tbody>
</table>

**5. Module Network Embedding**

<table border="1">
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
    </tbody>
    <tbody>
      <tr>
        <td rowspan=9><sub>Output</sub></td>
        <td><sub>embedding.csv</sub></td>
        <td><sub>network embedding</sub></td>
        <td><sub>:heavy_check_mark:</sub></td>
        <td><sub>:heavy_check_mark:</sub></td>
      </tr>
      <tr>
        <td><sub>plot_gene_cnt_each_cluster_Network.png</sub></td>
        <td><sub>number of gene per network module (same as the output in <code>Module Membership Analysis</code></sub></td>
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
        <td><sub>DEG distribution across modules (same as the output in <code>Module DE/Diagnostic Correlation</code></sub></td>
        <td><sub>:heavy_check_mark:</sub></td>
        <td><sub>:heavy_check_mark:</sub></td>
      </tr>
      <tr>
        <td><sub>cluster_DE_perc_epoch=100_alpha=0.1 embedding.png</sub></td>
        <td><sub>DEG distribution across embedding clusters (compare it with cluster_DE_perc_network.png)</sub></td>
        <td><sub>:heavy_check_mark:</sub></td>
        <td><sub>:heavy_check_mark:</sub></td>
      </tr>
      <tr>
        <td><sub>cluster_phenotype_corr_network.png</sub></td>
        <td><sub>module eigengene and alcohol trait correlation (same as the output in <code>Module DE/Diagnostic Correlation</code></sub></td>
        <td><sub>:heavy_check_mark:</sub></td>
        <td><sub></sub></td>
      </tr>
      <tr>
        <td><sub>cluster_phenotype_corr_embedding.png</sub></td>
        <td><sub>cluster eigengene and alcohol trait correlation (compare it with cluster_phenotype_corr_network.png)</sub></td>
        <td><sub>:heavy_check_mark:</sub></td>
        <td><sub></sub></td>
      </tr>
      <tr>
        <td><sub>alcohol trait correlation network vs embedding.png</sub></td>
        <td><sub>distribution plot to compare cluster_phenotype_corr_network.png and cluster_phenotype_corr_embedding.png</sub></td>
        <td><sub>:heavy_check_mark:</sub></td>
        <td><sub></sub></td>
      </tr>
      <tr>
        <td><sub>cluster_jaccard_Network vs epoch=100_alpha=0.1.png</sub></td>
        <td><sub>pairwise jaccard comparison to determine network module and embedding cluster similarity</sub></td>
        <td><sub>:heavy_check_mark:</sub></td>
        <td><sub>:heavy_check_mark:</sub></td>
      </tr>
    </tbody>
</table>

**6. ML and Critical Gene Identifier**

<table border="1">
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
    </tbody>
    <tbody>
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
    </tbody>
</table>
