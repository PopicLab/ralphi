## Deep reinforcement learning framework for haplotype assembly

##### Table of Contents  
[Overview](#overview)  
[Installation](#install)   
[User Guide](#guide)  
[Recommended workflow](#workflow)   

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
* Create new conda environment: ```$> conda create -n ralphi_env python=3.10 pip cxx-compiler```
* Activate the environment: ```$> conda activate ralphi_env```
* Install all the required packages: ```$> pip install -r install/requirements.txt```

* Set the ```PYTHONPATH``` as follows: ```export PYTHONPATH=${PYTHONPATH}:/path/to/ralphi```

<a name="guide"></a>
### User guide

#### Execution

To run: ```$> python engine/phase.py --config </path/to/config>```

```engine/phase.py``` phases an input VCF file given a pre-trained model and an input BAM file.
The script accepts as input a YAML file with configuration parameters. The ```config/``` directory provides 
a sample config file for the ONT platform (```config/ont.yaml```) and the Illumina platform (```config/illumina.yaml```),
which lists the required and key optional parameters.

The key required and optional YAML parameters for phasing are listed below:

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

#### Training Dataset Design

To be used to build a refined training and validation sets using filtering rules stated in dedicated yaml files.
To run:
```$> python engine/design_dataset.py --config </path/to/config>```
The key parameters of the config file are:

* ```panel``` [*required*] text file containing the paths to the BAM and VCF files to use for training. The BAM and its
corresponding VCF file have to be on the same line of the panel file, one pair per line.
Graphs and index dataframe will be saved in the parent folder of this text file.
* ```platform``` [*required*] sequencing platform (options: ```illumina``` or ```ONT```)
* ```filter_config``` [*optional*]: YAML file to specify selection filters to build the training and validation datasets.
If not provided, 200 random graphs will be selected fro validation and the rest for training.
* ```reference``` [*optional*, recommended for ONT] path to the reference FASTA file used for local realignment
* ```chr_names``` [*optional*] list of chromosomes to process: null (all) or a specific list e.g. ["chr1", "chr21"] (default: null)
* ```n_proc```  [*optional*] number of cores to use for training (default: 1)
* ```enable_read_selection```  [*optional*] enables Whatshap-based read downsampling (default: true for ONT)
* ```max_coverage```  [*optional*] target coverage for read downsampling (default: 15)
* ```mapq```  [*optional*] minimum alignment mapping quality, reads below this threshold will be removed (default: 20)
* ```filter_bad_reads```  [*optional*] remove low-quality highly-discordant reads (default: true for ONT)
* ```drop_chr``` [*optional*] list of strings representing chromosomes to hold out from training and validation (default: ['chr20'])

The syntax for the ```filter_config``` YAML relies on set of filters.
A filter involves a graph property and constraints on this property. The currently available graph properties are
```n_nodes```, ```n_edges```, ```max_weight```, ```min_weight```, ```diameter```, ```n_articulation_points```,
```density```, ```n_variants```, ```compression_factor```.
While the currently available constraints are ```min```, ```max``` and ```quantiles```. 
An example is provided below:
```yaml
# YAML TRAINING CONFIG FILE
shuffle: False
num_samples_per_category_default_train: 2000

global_filters:
  n_nodes:
    min: 10
    max: 5000
    
filter_categories:
  category_1:
    num_samples_train: 500
    filters:
      n_nodes: 
        max: 100
      n_edges:
        quantiles: [0.5, 0.8]
  category_2:
    num_samples_validate: 50
    filters:
      density:
        min: 0.1
        max: 0.5
```

The filters within the ```global_filters``` field will be used in order to filter out graphs.
From the remaining graphs, ```filter_categories``` is used to define groups of graphs. Each group has a name, a set of filters
and optionally a number of graphs to select ```num_samples_validate``` and/or ```num_samples_train```.
We enforce the different groups to be disjoint. Thus, the order of the group might be of importance as the 
graphs selected in ```category_1``` will not be available for ```category_2```.
If  ```num_samples_validate``` is not provided for a group, ```num_samples_per_category_default_validate``` will be used (default: 200).
If  ```num_samples_train``` is not provided for a group, ```num_samples_per_category_default_train``` will be used (default: all remaining graphs).
```shuffle``` (default: False) states if the graphs from the different groups should be shuffled together or if curriculum learning
should be used.

When using dataset_design.py, the graphs, the training and validation datasets will be cached in the folder containing the 
provided config file. A copy of the filter_config file will also be saved in this folder (if it was not provided, 
the default filter_config will be saved).

#### Training

Trained models are available under ```data/models```.
For users wanting to train their own model or fine-tune the provided models, they can run:
```$> python engine/train.py --config </path/to/config>```
The key parameters are:

* ```panel_dataset_train``` [*required*] list containing the paths to the dataset files to use for training. These files
are created using the dataset_design.py code (see section above). If several paths are provided the datasets 
will be concatenated in the paths order.
* ```panel_dataset_validate``` [*required*] list containing the paths to the dataset files to use for validation. These files
are created using the dataset_design.py code (see section above). If several paths are provided the datasets 
will be concatenated in the paths order.
* ```n_proc```  [*optional*] number of cores to use for training (default: 1)
* ```pretrained_model``` [*optional*] path to the pretrained ```ralphi``` model to fine-tune. If not provided, a new mode
will be initialized.
* ```gamma``` [*optional*] discount factor to be used for the computation of the loss function (default: 0.98)
* ```lr``` [*optional*] learning rate (default: $3e^{-5}$)
* ```epochs``` [*optional*] number of epochs to run (default: 1)



<a name="workflow"></a>
#### Recommended workflow 

1. Create a new directory.
2. Copy the appropriate YAML config from the ```config/``` folder into this directory.
3. Populate the YAML config file with the parameters specific to this experiment.
4. Run the ```phase.py``` script providing the path to the newly configured YAML file.

```ralphi``` will automatically create auxiliary directories with results in the folder where the config 
YAML files are located.
