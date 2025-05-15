from collections import defaultdict 
import logging
import operator

class Variant:
    """
    Heterozygous SNP in the VCF file
    """
    def __init__(self, vcf_idx):
        self.vcf_idx = vcf_idx
        self.phase_set = None
        self.frag_variants = []
        self.h = None
        self.phase_sets = defaultdict(list)

    def get_haplotype_support(self, frag_set):
        c0 = 0
        c1 = 0
        for v, n_copies in frag_set:
            if (v.haplotype == 0 and v.allele == '0') or (v.haplotype == 1 and v.allele == '1'):
                c0 += n_copies
            elif (v.haplotype == 0 and v.allele == '1') or (v.haplotype == 1 and v.allele == '0'):
                c1 += n_copies
        return c0, c1, c0 + c1

    def assign_haplotype(self):
        haplotype_support = {phase_set: self.get_haplotype_support(frag_set) for phase_set, frag_set in self.phase_sets.items()}
        # Check if we have equal evidence if there are multiple graphs covering
        if len(haplotype_support) > 1 and all(frag_copies == list(haplotype_support.values())[0][2] for _, _, frag_copies in haplotype_support.values()):
            self.h = None
            return
        self.phase_set = max(haplotype_support, key=lambda phase_set: haplotype_support[phase_set][2])
        c0, c1, _ = haplotype_support[self.phase_set]
        if c0 > c1:
            self.h = 0
        elif c0 < c1:
            self.h = 1
        else:
            self.h = None

    def __str__(self):
        return "Variant: vcf_idx={} frag_graph_idx={} depth={}".format(self.vcf_idx, self.phase_set,
                                                                       len(self.phase_sets))


def extract_variants(phased_frag_sets, chr):
    idx2variant = {}
    for ps, fragments in enumerate(phased_frag_sets):
        for frag in fragments:
            for var in frag.variants:
                vcf_idx = str(var.vcf_idx) + chr
                if vcf_idx not in idx2variant:
                    # first time we've seen this variant
                    idx2variant[vcf_idx] = Variant(var.vcf_idx)
                idx2variant[vcf_idx].phase_sets[ps].append((var, frag.n_copies))
    return idx2variant

def update_phase_sets(max_phase_set, idx2var):
    """
    For variants we do not phase (haplotype "h" field set to 2) we need to update the phase sets of variants downstream in the block
    since the block is split, and also need to handle the case of multiple unphased variants within a block.
    """
    sorted_vars = sorted(idx2var.items(), key=operator.itemgetter(0))
    change_set = False
    cur_phase_set = 0
    for var in sorted_vars:
        if change_set and var[1].phase_set != cur_phase_set:
            change_set = False
            max_phase_set += 1
        if change_set:
            if var[1].h is None: max_phase_set += 1
            else: var[1].phase_set = max_phase_set
        if var[1].h is None:
            cur_phase_set = var[1].phase_set
            change_set = True
    return dict(sorted_vars), max_phase_set

def split_eq_evidence(graphs, idx2var, max_phase_set=None, reads_num_th=50):
    if not max_phase_set: max_phase_set = len(graphs)
    else: max_phase_set = max_phase_set + 1
    for component in graphs:
        incident_variants_lookup = defaultdict(dict)
        vcf_positions = set()
        for frag_index, frag in enumerate(component):
            for var in frag.variants:
                vcf_positions.add(var.vcf_idx)
                incident_variants_lookup[frag_index][var.vcf_idx] = var
        vcf_positions = sorted(list(vcf_positions))
        for vcf_index, position in enumerate(vcf_positions):
            n_reads_spanning_var = total_evidence_var = n_alleles_same = n_alleles_diff = 0
            for frag_index, frag in enumerate(component):
                incident_variants = incident_variants_lookup[frag_index]
                if min(incident_variants) < position <= max(incident_variants):
                    n_reads_spanning_var += frag.n_copies
                    total_evidence_var += frag.n_copies
                    if (position not in incident_variants) or (position - 1 not in incident_variants): break
                    if incident_variants[position - 1].allele != incident_variants[position].allele:
                        n_alleles_diff += frag.n_copies
                    else:
                        n_alleles_same += frag.n_copies
            if (n_alleles_same + n_alleles_diff < total_evidence_var) or (not n_reads_spanning_var)\
                    or (n_reads_spanning_var > reads_num_th): continue
            if n_alleles_same == n_alleles_diff and n_alleles_same != 0:
                for i in vcf_positions[vcf_index:]:
                    if i in idx2var: idx2var[i].phase_set = max_phase_set
                max_phase_set = max_phase_set + 1

def postprocess(graphs, idx2var, config):
    if config.skip_post: return idx2var
    idx2var, max_phase_set = update_phase_sets(len(graphs), idx2var)
    split_eq_evidence(graphs, idx2var, max_phase_set)
    return idx2var

