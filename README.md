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
engine/design_dataset.py is used to generate fragment graph datasets from BAM and VCF files for model training and validation.
It supports both simple random sampling and fine-grained graph selection via configurable filters.

Usage:
```$> python engine/generate_dataset.py --config </path/to/config>```

Apart from the fragment and graph generation parameters mentioned earlier, the following configuration keys are available:
* ```panel``` [*required*] path to a text file listing BAM/VCF file pairs. 
Each line must contain the path to a BAM file and its corresponding VCF file, separated by whitespace or tab.
* ```platform``` [*required*] sequencing platform (options: ```illumina``` or ```ONT```).
* ```reference``` [*required* for ONT only] path to a FASTA file for local realignment.
* ```filter_config``` [*optional*]: YAML file defining filtering rules to construct training and validation sets.
If omitted, the pipeline selects ```num_samples_validate``` graphs for validation. The remainder is used for training.
* ```n_proc```  [*optional*] number of CPU cores for graph generation (default: 1).
* ```drop_chr``` [*optional*] list of chromosome to exclude (default: ['chr20']).
* ```num_samples_validate``` [*optional*]: number of graphs for validation (default: 200).
* ```num_samples_train``` [*optional*]: number of graphs for training. If null, all remaining graphs are used
(default: null).

##### Graph Filtering
To customize graph selection, users can provide an optional ```filter_config``` YAML file that defines filtering rules based on graph properties. 
Supported graph attributes are:
```n_nodes```, ```n_edges```, ```max_weight```, ```min_weight```, ```diameter```, ```n_articulation_points```, 
```density```, ```n_variants```, and ```compression_factor```.

Each property supports the following constraint types: ```min```, ```max``` and ```quantiles```. 

Refer to [FILTER CONFIG](docs/filter_config.yaml) for an example.

###### Global Filters
Defined in the ```global_filters``` section—these filters eliminate graphs before any category selection.

###### Filter Categories
Defined in the ```filter_categories``` section. 

Each category includes:
* Name: used to track performance in Weights & Biases .
* Filters: a set of graph metric-based rules.
* Optional:
  * ```num_samples_validate_category``` 
  * ```num_samples_train_category```.

**Important:** Categories are mutually exclusive and order matters—once graphs are assigned to a category, 
they are removed from subsequent categories.

Fallback behaviors:
* If  ```num_samples_validate_category``` is unspecified, ```num_samples_validate``` is used.
* If  ```num_samples_train_category``` is unspecified, ```num_samples_train``` is used.
* ```shuffle``` (default: False) determines whether to apply curriculum learning or randomize sample order.

##### Output
All outputs are saved in the directory of the provided config file:
* Training and Validation datasets.
* Copies of the ```filter_config``` (if originally provided) and ```panel```.
* Cached graphs in an ```output/``` subdirectory.

#### Training

Trained models are available in the ```data/models``` directory.

To train or fine-tune models:
```$> python engine/train.py --config </path/to/config>```

##### Configuration Keys
* ```panel_dataset_train``` [*required*] list of paths to training datasets. Multiple paths are concatenated in order.
* ```panel_dataset_validate``` [*required*] list of paths to validation datasets. Multiple paths are concatenated in order.
* ```n_proc```  [*optional*] number of CPU cores to use for validation (default: 1)
* ```pretrained_model``` [*optional*] path to a pretrained ```ralphi``` model for fine-tuning. If omitted, training starts
from scratch.
* ```gamma``` [*optional*] discount factor used in the loss function (default: 0.98).
* ```lr``` [*optional*] learning rate (default: $3e^{-5}$).
* ```epochs``` [*optional*] number of training epochs (default: 1).
* ```device``` [*optional*] compute device, ```cpu``` or ```cuda``` for gpu (default: ```cpu```).
* ```interval_episodes_to_validation``` [*optional*] number of training episodes between validation runs (default: 500).
* ```log_wandb``` [*optional*] Whether to log training/validation metrics to Weights & Biases  (default: False).
* ```project_name``` [*optional*] Weights & Biases  project name (default: ralphi).
* ```run_name``` [*optional*] name for this Weights & Biases run (default: ralphi). 

##### Output
* Intermediate model snapshots (at each validation checkpoint) in an ```output/``` subdirectory.
* The model with the highest validation reward in ```output/ralphi_model_best.pt```.



<a name="workflow"></a>
#### Recommended workflow 

1. Create a new directory.
2. Copy the appropriate YAML config from the ```config/``` folder into this directory.
3. Populate the YAML config file with the parameters specific to this experiment.
4. Run the ```phase.py``` script providing the path to the newly configured YAML file.

```ralphi``` will automatically create auxiliary directories with results in the folder where the config 
YAML files are located.
