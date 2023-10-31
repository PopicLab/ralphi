import argparse


class FragVariant:
    def __init__(self, vcf_idx, allele, qscore=None, n_copies=1):
        self.vcf_idx = vcf_idx
        self.allele = allele  # 0 or 1
        self.qscore = qscore
        self.n_copies = n_copies

        # phasing metadata
        self.haplotype = None

    def __str__(self):
        return "Variant: vcf_idx={} allele={} qscore={} n_copies={}".format(self.vcf_idx, self.allele,
                                                                            self.qscore, self.n_copies)

    def __eq__(self, v):
        # temporarily disable equality check by qscore, since on simulated data this is not meaningful for compression
        #return (self.vcf_idx, self.allele, self.qscore, self.haplotype) == (v.vcf_idx, v.allele, v.qscore, v.haplotype)
        return(self.vcf_idx, self.allele, self.haplotype) == (v.vcf_idx, v.allele, v.haplotype)


class Block:
    def __init__(self, vcf_idx, alleles, qscores=None):
        self.vcf_idx_start = vcf_idx
        self.vcf_idx_end = vcf_idx + len(alleles) - 1
        self.variants = []
        for i in range(len(alleles)):
            qscore = None if (qscores is None) else qscores[i]
            self.variants.append(FragVariant(vcf_idx + i, alleles[i], qscore))
        self.n_variants = len(self.variants)

    def __str__(self):
        return "Block: n_variants={} vcf_idx={} variants=\n".format(self.n_variants, self.vcf_idx_start) \
               + '\n'.join(map(str, self.variants))

    def __eq__(self, block):
        return (self.vcf_idx_start, self.vcf_idx_end, self.n_variants, self.variants) == (
            block.vcf_idx_start, block.vcf_idx_end, block.n_variants, block.variants)

    def overlaps(self, block):
        return (min(self.vcf_idx_end, block.vcf_idx_end) - max(self.vcf_idx_start, block.vcf_idx_start)) >= 0


class Fragment:
    def __init__(self, read_name, blocks=None, variants=None):
        self.read_id = read_name
        self.n_copies = 1  # number of copies of this fragment
        self.n_variants = 0
        self.variants = []  # flat list of all variants
        self.vcf_idx_start = None
        self.vcf_idx_end = None
        if blocks:
            self.vcf_idx_start = blocks[0].vcf_idx_start
            self.vcf_idx_end = blocks[len(blocks)-1].vcf_idx_end
            for block in blocks:
                self.n_variants += block.n_variants
                self.variants.extend(block.variants)
        elif variants:
            self.variants = variants
            self.vcf_idx_start = variants[0].vcf_idx
            self.vcf_idx_end = variants[len(variants)-1].vcf_idx
            self.n_variants += len(self.variants)
        self.quality = [v.qscore for v in self.variants]

        self.strand = None
        self.read_barcode = None
        self.haplotype = None
        self.true_haplotype = None
        self.vcf_positions = None

    def __eq__(self, fragment):
        return (self.vcf_idx_start, self.vcf_idx_end, self.n_variants, self.variants) == (
            fragment.vcf_idx_start, fragment.vcf_idx_end, fragment.n_variants, fragment.variants)

    def __str__(self):
        return "\nread_id={} n_variants={} vaariants=\n".format(self.read_id, self.n_variants) + \
               '\n'.join(map(str, self.variants))

    def overlap(self, fragment):
        if min(self.vcf_idx_end, fragment.vcf_idx_end) < max(self.vcf_idx_start, fragment.vcf_idx_start):
            return []
        shared_variants = []
        for v1 in self.variants:
            for v2 in fragment.variants:
                if v1.vcf_idx == v2.vcf_idx:
                    shared_variants.append((v1, v2))
        return shared_variants

    def assign_haplotype(self, h):
        self.haplotype = h
        for var in self.variants:
            var.haplotype = h

    @staticmethod
    def parse_from_file(frag_line):
        fields = frag_line.split()
        n_blocks = int(fields[0])
        field_idx = 2
        blocks = []
        for _ in range(n_blocks):  # parse each block
            vcf_index = int(fields[field_idx]) - 1  # -1 since the variant index is 1-based in the fragment file
            alleles = fields[field_idx + 1]
            field_idx += 2
            blocks.append(Block(vcf_index, alleles))
        # parse quality scores for all variants
        qscores = fields[field_idx]
        qscore_idx = 0
        for block in blocks:
            for variant in block.variants:
                variant.qscore = 10 ** ((ord(qscores[qscore_idx]) - 33) / (-10))
                qscore_idx += 1
        return Fragment(fields[1], blocks=blocks)


def parse_frag_file(frags_fname):
    fragments = []
    with open(frags_fname, 'r') as f:
        for frag_line in f:
            fragments.append(Fragment.parse_from_file(frag_line))
    # sort fragments to optimize graph construction
    fragments = sorted(fragments, key=lambda frag: frag.vcf_idx_start)
    return fragments


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load fragment file")
    parser.add_argument('--frags', help='Input fragment file')
    args = parser.parse_args()
    fragments = parse_frag_file(args.frags)

