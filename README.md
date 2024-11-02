## Deep reinforcement learning framework for haplotype assembly

### Overview

```ralphi``` is a deep reinforcement learning framework for haplotype
assembly, which leverages a graph convolutional neural network and an actor-critic 
reinforcement learning model to learn the read-to-haplotype assignment algorithm 
based on the maximum fragment cut (MFC) objective. 

<a name="install"></a>
### Installation

* Clone the repository:  ```$> git clone git@github.com:PopicLab/ralphi.git```

* ```$> cd ralphi```

* Create a new virtual environment: ```$> python3.10 -m venv env```

* Activate the environment: ```$> source env/bin/activate```

* Install all the required packages: ```$> pip install -r requirements.txt```

* Set the ```PYTHONPATH``` as follows: ```export PYTHONPATH=${PYTHONPATH}:/path/to/ralphi```

<a name="guide"></a>
### Execution

To run: ```$> python engine/phase.py --config </path/to/config>```

```engine/phase.py``` phases an input VCF file given a pre-trained model and an input BAM file.
The key required and optional YAML parameters for phasing are listed below:

* ```vcf``` [*required*] path to the VCF variant file (must be compressed and indexed)
* ```bam``` [*required*] path to the BAM alignment file 
* ```platform``` [*required*] sequencing platform (options: ```illumina``` or ```ONT```) 
* ```reference``` [*required*] path to the referene FASTA file
* ```model_path``` [*required*] path to the pretrained ralphi model (available under )
* ```chr_names``` [*optional*] list of chromosomes to process: null (all) or a specific list e.g. ["chr1", "chr21"] (default: null)
* ```n_proc```  [*optional*] number of cores to use for phasing (default: 1)

Two models are currently provided in the ```data/models``` directory: 
(1) ralphi.v1.long.pt is recommended for ONT inputs and (2) ralphi.v1.short.pt is recommended for Illumina short-read inputs.
