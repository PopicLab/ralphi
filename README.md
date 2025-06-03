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

This module enables the construction of training and validation sets. It supports filters
specified in a YAML configuration file for a refined selection.
To run:
```$> python engine/design_dataset.py --config </path/to/config>```
In addition to the parameters for the fragments and graphs generation listed in the previous section, the  
config key parameters are:
* ```panel``` [*required*] text file containing the paths to the BAM and VCF files for training and validation. 
Each line should contain one BAM and its corresponding VCF file.
* ```platform``` [*required*] sequencing platform (options: ```illumina``` or ```ONT```).
* ```reference``` [*required* only for ONT] path to the reference FASTA file used for local realignment.
* ```filter_config``` [*optional*]: YAML file to specify selection filters to build the training and validation datasets.
If omitted, 200 graphs will be randomly selected for validation, and the rest for training.
* ```n_proc```  [*optional*] number of cores for graph generation (default: 1).
* ```drop_chr``` [*optional*] list of chromosomes to exclude from training and validation datasets (default: ['chr20']).
* ```num_samples_validate``` [*optional*]: number of graphs to retain for validation (default: 200).
* ```num_samples_train``` [*optional*]: number of graphs to retain for training, if null, all remaining graphs will be selected
(default: null).

The filter_config YAML defines filters based on graph properties such as:
```n_nodes```, ```n_edges```, ```max_weight```, ```min_weight```, ```diameter```, ```n_articulation_points```,
```density```, ```n_variants```, ```compression_factor```.

Using the constraints: ```min```, ```max``` and ```quantiles```. 

Refer to the [FILTER CONFIG](docs/filter_config.yaml) example for structure and syntax.

The filters within the ```global_filters``` section filter out graphs prior to dataset assignment.

The ```filter_categories``` section defines categories of graphs. 

Each category includes:
* A name: which will be used in wandb to visualize the performance per category of graphs.
* A set of filters.
* [*optional*] ```num_samples_validate_category``` and/or ```num_samples_train_category```.

Each category is mutually exclusive. Order matters—graphs selected for one category are unavailable for the next.
If  ```num_samples_validate_category``` is missing, the default (```num_samples_validate```) is used.
If  ```num_samples_train_category``` is missing, the default (```num_samples_train```) is used.
The ```shuffle``` (default: False) determines whether to use curriculum learning or mix samples randomly.

Outputs in the directory containing the config file:
* Training/Validation datasets.
* Copies of the ```filter_config``` (if provided) and ```panel``` files.
* Cached graphs are stored in an ```output``` subdirectory.

#### Training

Trained models are available in the ```data/models``` directory.
To train or fine-tune models:
```$> python engine/train.py --config </path/to/config>```

Key training parameters:
* ```panel_dataset_train``` [*required*] list of paths to training datasets. Multiple paths are concatenated in order.
* ```panel_dataset_validate``` [*required*] list of paths to validation datasets. Multiple paths are concatenated in order.
* ```n_proc```  [*optional*] number of cores to use for validation (default: 1)
* ```pretrained_model``` [*optional*] path to a pretrained ```ralphi``` model for fine-tuning. If omitted, training starts
from scratch.
* ```gamma``` [*optional*] discount factor for loss computation (default: 0.98).
* ```lr``` [*optional*] learning rate (default: $3e^{-5}$).
* ```epochs``` [*optional*] number of training epochs (default: 1).
* ```device``` [*optional*] execution device, either ```cpu``` or ```cuda``` for gpu (default: ```cpu```).
* ```interval_episodes_to_validation``` [*optional*] number of episodes between validation steps (default: 500).
* ```log_wandb``` [*optional*] if the training and validation performance have to be sent to wandb (default: False).
* ```project_name``` [*optional*] name of the project in wandb (default: ralphi).
* ```run_name``` [*optional*] name of this specific run in the project (default: ralphi). 

The models at each validation step will be saved in an ```output``` subdirectory.
The model presenting the best validation reward is named ```ralphi_model_best.pt```.



<a name="workflow"></a>
#### Recommended workflow 

1. Create a new directory.
2. Copy the appropriate YAML config from the ```config/``` folder into this directory.
3. Populate the YAML config file with the parameters specific to this experiment.
4. Run the ```phase.py``` script providing the path to the newly configured YAML file.

```ralphi``` will automatically create auxiliary directories with results in the folder where the config 
YAML files are located.
