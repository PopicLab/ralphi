## Deep reinforcement learning framework for haplotype assembly

##### Table of Contents  
[Overview](#overview)  
[Installation](#install)   
[User Guide](#guide)  
&nbsp;&nbsp;  [Phasing](#phase)   
&nbsp;&nbsp;  [Training data generation](#generate)   
&nbsp;&nbsp;  [Training](#train)   
 

<a name="overview"></a>
### Overview

```ralphi``` is a deep reinforcement learning framework for haplotype
assembly, which leverages a graph convolutional neural network and an actor-critic 
reinforcement learning model to learn the read-to-haplotype assignment algorithm 
based on the maximum fragment cut (MFC) objective. 

<p align="center">
<img src="docs/img/ralphi_overview.png" alt="drawing" width="1000"/>
</p>

<a name="install"></a>
### Installation

* Clone the repository:  ```$> git clone git@github.com:PopicLab/ralphi.git```
* ```$> cd ralphi```
* Create new conda environment: ```$> conda create -n ralphi_env python=3.10 pip cxx-compiler bcftools=1.21```
* Activate the environment: ```$> conda activate ralphi_env```
* Install all the required packages: ```$> pip install -r install/requirements.txt```

* Set the ```PYTHONPATH``` as follows: ```export PYTHONPATH=${PYTHONPATH}:/path/to/ralphi```

<a name="guide"></a>
### User guide

<a name="phase"></a>
#### Execution

To run: ```$> python engine/phase.py --config </path/to/config>```

```phase.py``` phases an input VCF file given a pre-trained model and an input BAM file.
The script accepts as input a YAML file with configuration parameters. The ```config/``` directory provides 
a sample config file for the ONT platform (```config/ont.yaml```) and the Illumina platform (```config/illumina.yaml```),
which lists the required and key optional parameters.

The key parameters for phasing are listed below:

* ```vcf``` [*required*] path to the VCF variant file (must be compressed and indexed)
* ```bam``` [*required*] path to the BAM alignment file 
* ```platform``` [*required*] sequencing platform (options: ```illumina``` or ```ONT```) 
* ```reference``` [*required* only for ONT] path to the reference FASTA file used for local realignment
* ```model``` [*required*] path to the pretrained ```ralphi``` model (available under ```data/models```)
* ```chr_names``` [*optional*] list of chromosomes to process: null (all) or a specific list e.g. ["chr1", "chr21"] (default: null)
* ```n_proc```  [*optional*] number of cores to use for phasing (default: 1)
* ```enable_read_selection```  [*optional*] enables Whatshap-based read downsampling (default: true for ONT)
* ```max_coverage```  [*optional*] target coverage for read downsampling (default: 15)
* ```mapq```  [*optional*] minimum alignment mapping quality, reads below this threshold will be removed (default: 20)
* ```filter_bad_reads```  [*optional*] remove low-quality highly-discordant reads (default: true for ONT)

Two models are currently provided in the ```data/models``` directory: 
(1) ralphi.v1.long.pt is recommended for ONT inputs and (2) ralphi.v1.short.pt is recommended for Illumina 
short-read inputs.

```ralphi``` will output results in the parent directory of the YAML config file. The results include the phased VCF
files for each input chromosome (under ```output/```; e.g. ```output/chr1.ralphi.vcf```) and execution logs.

<a name="workflow"></a>
#### Recommended workflow 

1. Create a new directory.
2. Copy the appropriate YAML config from the ```config/``` folder into this directory.
3. Populate the YAML config file with the parameters specific to this experiment.
4. Run the ```phase.py``` script providing the path to the newly configured YAML file.

```ralphi``` will automatically create auxiliary directories with results in the folder where the config 
YAML files are located.

<a name="generate"></a>
#### Training dataset generation
```generate_dataset.py``` can be used to generate fragment graphs from BAM and VCF files for model training and validation.
It supports both random graph sampling and fine-grained graph selection via configurable filters.

Usage:
```$> python engine/generate_dataset.py --config </path/to/config>```

The key parameters for graph dataset generation are listed below:
* ```panel``` [*required*] path to a TSV file with BAM/VCF file pairs; each line must contain the path to the BAM file and its corresponding VCF file
* ```platform``` [*required*] sequencing platform (options: ```illumina``` or ```ONT```) 
* ```reference``` [*required*] path to the reference FASTA file 
* ```selection_config``` [*optional*]: YAML file defining graph selection criteria
* ```n_proc```  [*optional*] number of CPU cores to use for graph generation (default: 1)
* ```drop_chr``` [*optional*] list of chromosome to exclude from the dataset (default: ['chr20'])
* ```size``` [*optional*]: maximum number of graphs to include iff selection_config is not provided; if null, all the graphs will be included (default: null)
* ```validation_ratio``` [*optional*]: fraction of graphs to select for validation (default: 0.1)
(default: null)

Note: the following parameters defined for phasing (see above) are also available for training graph generation: 
```enable_read_selection```, ```max_coverage```, ```mapq```, and ```filter_bad_reads```.

##### Graph selection
Users can provide a YAML file that defines graph selection criteria based on the following graph properties:
* ```n_nodes```: number of graph nodes
* ```n_edges```: number of graph edges
* ```max_weight```: maximum edge weight 
* ```min_weight```: minimum edge weight
* ```diameter```: graph diameter 
* ```n_articulation_points```: number of articulation points 
* ```density```: graph density 
* ```n_variants```: number of variants covered by the graph

For each property, the user can either provide a range of allowed values using the ```min``` and/or ```max``` parameters 
or use the ```quantiles``` parameter to select graphs that fall within the specified quantile range.
Graph selection rules can be applied globally to all graphs (using the ```Global``` config section) or to define 
specific graph subsets. Global selection rules are applied prior to any additional selection criteria.

Multiple graph subsets can be defined in the config, under a unique name (used for performance tracking). 
Each subset can specify multiple selection criteria, as well as optional ```size_validate``` and ```size_train``` 
parameters, which specify how many such graphs to include into this subset for validation and training, respectfully.
If ```size_train``` is not specified, all the qualifying graphs will be selected. If ```size_validate``` is not specified, 
the default ```validation_ratio``` will be used to determine the number of graphs to select for validation. 

**Important:** Graph subsets are mutually exclusive — once graphs are assigned to a subset, they are removed from 
subsequent selection. The ```shuffle``` parameter can be used to specify in what order the graphs should be processed. 
If ```shuffle``` is set to ```True```, the selected graphs will be randomly shuffled in the final dataset; otherwise
the graphs will be ordered according to the order in which their subsets are defined in the config file (enabling curriculum learning). 

Please refer to the [sample graph selection config](config/graph_selection_example.yaml) for an example.

All outputs will be written to the directory of the provided config file, including all the generated graphs 
(in the ```output/``` subdirectory) and the training and validation dataset indices 
(```train.graph_index``` and ```validate.graph_index```) which can be provided for model training.

<a name="train"></a>
#### Training
```train.py``` can be used to train or fine-tune a ```ralphi``` model.

Usage:
```$> python engine/train.py --config </path/to/config>```

The key parameters for model training are listed below:
* ```panel_dataset_train``` [*required*] list of paths to the training dataset indices (```.graph_index``` files)
* ```panel_dataset_validate``` [*required*] list of paths to the validation dataset indices (```.graph_index``` files)
* ```pretrained_model``` [*optional*] path to a pretrained ```ralphi``` model for fine-tuning
* ```n_proc```  [*optional*] number of CPU cores to use for validation (default: 1)
* ```gamma``` [*optional*] discount factor used in the loss function (default: 0.98).
* ```lr``` [*optional*] learning rate (default: $3e^{-5}$).
* ```epochs``` [*optional*] number of training epochs (default: 1).
* ```device``` [*optional*] compute device, ```cpu``` or ```cuda``` for gpu (default: ```cpu```).
* ```interval_episodes_to_validation``` [*optional*] number of training episodes between validation runs (default: 500).

The model with the highest validation reward (```ralphi_model_best.pt```), along with all intermediate model snapshots 
(at each validation checkpoint), will be written to the ```output``` directory created in the folder of the provided 
YAML config file.
