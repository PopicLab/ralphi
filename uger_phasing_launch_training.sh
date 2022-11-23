#!/bin/bash

#############################
### Default UGER Requests ###
#############################

# This section specifies uger requests.  
# This is good for jobs you need to run multiple times so you don't forget what it needs.

# Memory request for 4G
#$ -l h_vmem=20G

# I like single output files
#$ -j y

# Runtime request.  Usually 30 minutes is plenty for me and helps me get backfilled into reserved slots.
#$ -l h_rt=170:08:00

# I don't like the top level of my homedir filling up.
#$ -o /broad/popiclab/data/phasing/experiments/BroadClusterPrintingOutput/

######################
### Dotkit section ###
######################

# This is required to use dotkits inside scripts
source /broad/software/scripts/useuse

# Use your dotkit
reuse Python-3.9
cd /home/unix/amaheshw/PhasingProject/dphase
source env_with_39_python/bin/activate
export PYTHONPATH=/home/unix/amaheshw/PhasingProject/dphase
##################
### Run script ###
##################

python engine/train.py --config $1
