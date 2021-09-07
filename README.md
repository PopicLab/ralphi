## RL-based genome phasing

### Overview

Genome phasing is a classic key problem in genome analysis that involves partitioning read fragments into the maternal 
and paternal haplotype sets. Reads that span more than one variant position provide evidence about which alleles 
occur on the same haplotype; however, read errors make this problem challenging. 
Current read-based phasing methods heuristically partition the fragments using a variety of metrics, 
for example minimizing the number of errors that need to be corrected in each set 
(the NP-hard MEC objective used by HapCut) or solving a max-cut problem (the NP-hard MFC objective used by RefHap). 
In this project, we investigate a new approach to genome phasing based on reinforcement learning. 
Our RL agent acts by selecting and assigning a fragment to a haplotype given the MFC-based reward and a state 
represented by a graph convolutional network, which embeds the read fragment overlap graph derived 
from the input dataset. 

### Code execution

Clone the repository with its submodules:

```git clone --recursive git@github.com:PopicLab/dphase.git```

Prior to running the code, setup a Python virtual environment (as described below) 
and set the PYTHONPATH as follows: ```export PYTHONPATH=${PYTHONPATH}:${PATH_TO_REPO}```

The ```engine``` directory contains the following key scripts to train/evaluate the model:

* ```train.py```: trains the phasing network given a panel of fragment files
* ```test.py```: phases an input VCF file given a pre-trained model and an input fragment file 

Execution:

```python engine/train.py --panel=/path/to/panel --out_dir=/path/to/output --max_episodes=N```

```python engine/test.py --panel=/path/to/test/panel --model=/path/to/model --input_vcf=/path/to/input_vcf --output_vcf=/path/to/output_phased_vcf```


### Input fragment file generation

A per-chromosome fragment file is generated from an unphased VCF and BAM file pair using the third-party 
HapCUT2 ```extractHAIRS``` script. 
The WDL workflow to produce such fragment files is implemented in ```workflows/frags.wdl```.
The inputs to this workflow are: a TSV panel of samples (which includes sample names and bam file paths), 
a list of chromosomes, and a path to a VCF directory 
(see the example input parameter file: ```workflows/frags.json```). 
The output of the workflow is a set of fragment files -- one for each sample and chromosome specified
in the input. See ```seq/frags.py``` for information about the format of fragment files.

To execute the workflow using WDL/Cromwell:
```java -jar cromwell-51.jar run workflows/frags.wdl --inputs workflows/frags.json```

### Fragment graphs and phasing environment basics

A fragment graph (```FragGraph```) is initialized given a fragment file: nodes represent read fragments 
spanning >= 2 variants and edges connect fragments that overlap (i.e. have some variants in common). 
The weight of each edge is computed based on the number of shared variants and the number of conflicting alleles 
(see ```graphs/frag_graph.py```). Fragment graphs are decomposed into connected components prior to phasing
and represent the main state of the phasing environment (embedded using DGL). Each node is associated with 
a binary feature indicating whether the node has been selected as part of the solution 
(additional features are also explored).

The phasing environment state consists of the fragment graph and the node assignments to each haplotype (0: H0 1: H1).
The action space consists of the set of all graph nodes (i.e. the assignment of a node to a haplotype) 
and a termination step. MFC score is used as the reward function. 

### Benchmarking

The third-party evaluation script ```calculate_haplotype_statistics.py``` provided by HapCUT2 is used for evaluation
and benchmarking. The script can be executed as follows:

```python3 third-party/HapCUT2/utilities/calculate_haplotype_statistics.py -v1 /path/to/output -v2 /path/to/ground_truth```


#### Setup Python virtual environment (recommended)

1. Create the virtual environment (in the venv directory): 
```$> python3.7 -m venv env```

2. Activate the environment: 
```$> source env/bin/activate```

3. Install all the required packages (in the virtual environment):
```$> pip --no-cache-dir install -r ../requirements.txt```
