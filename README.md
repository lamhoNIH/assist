# ASSIST (Alcoholism Solutions: Synthesizing Information to Support Treatments)

This repo contains the source code of ASSIST analysis modules developed under the Phase I of the ASSIST project. These modules can be run in Jupyter notebooks, Docker containers, as well as workflows managed by the ADE (Active Discovery Engine) with increasing support for provenance.

### Description of analysis modules:
| Analysis Module | Description |
|-------|-------------|
| Network Analysis | Network construction from expression data and prepping network + phenotypic trait file |
| Module Extraction | Network module detection by Louvain Algorithm |
| Module Membership Analysis | Genes count per module and module assignment stability check |
| Module DE/Diagnostic Correlation | Biological relevance of network module check |
| Module Network Embedding | Conversion of network to machine learning-friendly matrix representation |
| ML and Critical Gene Identifier | Machine learning to extract features for critical gene identification |

The above analysis modules are inter-related as depicted in the conceptual workflow below.
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

**Network Analysis**

**Module Extraction**

**Module Membership Analysis**

**Module DE/Diagnostic Correlation**

**Module Network Embedding**

**ML and Critical Gene Identifier**
