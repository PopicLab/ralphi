#!/bin/bash

#############################
### Default UGER Requests ###
#############################

# This section specifies uger requests.  
# This is good for jobs you need to run multiple times so you don't forget what it needs.

# Memory request for 4G
#$ -l h_vmem=2G

# Cores
#$ -pe smp 4
#$ -binding linear:4

# I like single output files
#$ -j y

# Runtime request.  Usually 30 minutes is plenty for me and helps me get backfilled into reserved slots.
#$ -l h_rt=00:08:00

# I don't like the top level of my homedir filling up.
#$ -o $HOME/outputsPhasing/

######################
### Dotkit section ###
######################

# This is required to use dotkits inside scripts
source /broad/software/scripts/useuse

# Use your dotkit
reuse Python-3.6

##################
### Run script ###
##################
cd /home/unix/amaheshw/PhasingProject/dphase
python test.py --panel $1
