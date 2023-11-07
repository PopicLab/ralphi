from collections import defaultdict 

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
