
class Variant:
    """
    Heterozygous SNP in the VCF file
    """
    def __init__(self, vcf_idx):
        self.vcf_idx = vcf_idx

        # phasing metadata
        self.phase_set = None
        self.frag_variants = []
        self.h = None

    def assign_haplotype(self):
        c0 = 0
        c1 = 0
        for v, n_copies in self.frag_variants:
            if (v.haplotype == 0 and v.allele == '0') or (v.haplotype == 1 and v.allele == '1'):
                c0 += n_copies
            elif (v.haplotype == 0 and v.allele == '1') or (v.haplotype == 1 and v.allele == '0'):
                c1 += n_copies
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
    for i, fragments in enumerate(phased_frag_sets):
        for frag in fragments:
            for var in frag.variants:
                if var.vcf_idx not in idx2variant:
                    idx2variant[var.vcf_idx] = Variant(var.vcf_idx)
                idx2variant[var.vcf_idx].phase_set = i
                idx2variant[var.vcf_idx].frag_variants.append((var, frag.n_copies))
    return idx2variant

