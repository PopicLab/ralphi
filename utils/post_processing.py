from collections import defaultdict
import operator

def update_split_block_phase_sets(solutions, idx2var):
    """
    For variants we do not phase (haplotype "h" field set to 2) we need to update the phase sets of variants downstream in the block
    since the block is now split, and also need to handle the case of multiple unphased variants within a block.
    """
    max_phase_set = len(solutions)
    sorted_vars = sorted(idx2var.items(), key=operator.itemgetter(0))
    change_set = False
    cur_phase_set = 0
    for i in sorted_vars:
        if change_set and i[1].phase_set != cur_phase_set: 
            # end of current block
            change_set = False
            max_phase_set += 1
        if change_set:
            if i[1].h == None: 
                # hit another unphased variant in the same block, so need to change phase set again
                max_phase_set += 1
            else:
                i[1].phase_set = max_phase_set
        if i[1].h == None: 
            # hit a variant that shouldn't be phased, so change the phase set of subsequent variants
            cur_phase_set = i[1].phase_set
            change_set = True
    return dict(sorted_vars), max_phase_set

def split_low_evidence(solutions, idx2var, max_phase_set=None, evidence_threshold=0):
    """
    Args:
        solutions: list of list of phased fragments, where each index corresponds to a connected component
        idx2var: phased variant dictionary, used to generate the output phasing result per variant
        max_phase_set: current max phase set (from previous post-processing steps
        evidence_threshold: amount of direct evidence across consecutive reads required in order to confidently phase
        (default 0 corresponds to only splitting equal evidence cases)
    """
    if not max_phase_set:
        max_phase_set = len(solutions)
    else:
        max_phase_set = max_phase_set + 1

    for component in solutions:
        # for some specific graph, we are enumerating evidence across pairs of consecutive variants
        incident_variants_lookup = defaultdict(dict)
        vcf_positions = set()
        for frag_index, frag in enumerate(component):
            for var in frag.variants:
                vcf_positions.add(var.vcf_idx)
                incident_variants_lookup[frag_index][var.vcf_idx] = var
        vcf_positions = sorted(list(vcf_positions))

        for vcf_index, position in enumerate(vcf_positions):
            n_reads_spanning_var = n_alleles_same = n_alleles_diff = 0

            for frag_index, frag in enumerate(component):
                incident_variants = incident_variants_lookup[frag_index]
                if min(incident_variants) < position <= max(incident_variants):
                    n_reads_spanning_var += frag.n_copies

                    if (position not in incident_variants) or (position - 1 not in incident_variants):
                        # considering direct evidence caused by consecutive phased variants
                        break
                    if incident_variants[position - 1].allele != incident_variants[position].allele:
                        n_alleles_diff += frag.n_copies
                    else:
                        n_alleles_same += frag.n_copies

            if (n_alleles_same + n_alleles_diff < n_reads_spanning_var) or (n_reads_spanning_var == 0):
                # more complex case with non-consecutive variants being covered by evidence reads;
                # not always ambiguous depending on structure of other reads around these variants
                continue
            if abs(n_alleles_same - n_alleles_diff) <= evidence_threshold and n_alleles_same and n_alleles_diff:
                # ambiguous!!
                for i in vcf_positions[vcf_index:]:
                    idx2var[i].phase_set = max_phase_set
                max_phase_set = max_phase_set + 1

def postprocess_ambiguous(solutions, idx2var, evidence_threshold=0):
    idx2var, max_phase_set = update_split_block_phase_sets(solutions, idx2var)
    split_low_evidence(solutions, idx2var, max_phase_set, evidence_threshold=evidence_threshold)
    return idx2var