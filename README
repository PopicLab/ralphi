## RL-based genome phasing

### Overview

Genome phasing is a classic key problem in genome analysis that involves partitioning read fragments into the maternal and paternal haplotype sets. 
Reads that span more than one variant position provide evidence about which alleles occur on the same haplotype; however, read errors make 
this problem challenging. Current read-based phasing methods heuristically partition the fragments using a variety of metrics, 
for example minimizing the number of errors that need to be corrected in each set (the NP-hard MEC objective used by HapCut) 
or solving a max-cut problem (the NP-hard MFC objective used by RefHap). 
In this project, we investigate a new approach to genome phasing based on reinforcement learning. Our RL agent acts by selecting and assigning a fragment 
to a haplotype given the MFC-based reward and a state represented by a graph convolutional network (which embeds the read fragment overlap graph derived 
from the input dataset). 

