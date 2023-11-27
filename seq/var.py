from collections import defaultdict 
import logging

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
        
        # Check if all we have equal evidence if there are multiple graphs covering 
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
                                                                       len(self.phase_set))

def flip_phase(h):
    if h == None:
        return h
    return (h + 1) % 2

def stitch_specific(frag_list, join, lookup_component):
    first_hap = join[0][1].haplotype
    first_index = lookup_component[join[0][0]]
    for comp_index, frag in join[1:]:
        if frag.haplotype != first_hap:
            for frag_to_flip in frag_list[lookup_component[comp_index]]:
                frag_to_flip.assign_haplotype(flip_phase(frag_to_flip.haplotype))
        frag_list[first_index].extend(frag_list[lookup_component[comp_index]])
        frag_list[lookup_component[comp_index]] = []
        lookup_component[comp_index] = first_index
    return frag_list, lookup_component


def stitch_fragments(solutions):
    logging.info("Stitching fragments...")
    lookup_stitch = {}
    lookup_component = {i: i for i in range(len(solutions))}
    for component_index, frag_list in enumerate(solutions):
        for frag in frag_list:
            if frag.fragment_group_id is not None:
                lookup_stitch.setdefault(frag.fragment_group_id, []).append((component_index, frag))
    condensed_components = solutions
    for elem in lookup_stitch:
        condensed_components, lookup_component = stitch_specific(condensed_components, lookup_stitch[elem], lookup_component)
    logging.info("Finished stitching fragments...")
    return condensed_components

def extract_variants(phased_frag_sets):
    idx2variant = {}
    for ps, fragments in enumerate(phased_frag_sets):
        for frag in fragments:
            for var in frag.variants:
                if var.vcf_idx not in idx2variant:
                    # first time we've seen this variant
                    idx2variant[var.vcf_idx] = Variant(var.vcf_idx)
                idx2variant[var.vcf_idx].phase_sets[ps].append((var, frag.n_copies))
    return idx2variant
