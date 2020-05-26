import argparse


class VariantHandle:
    def __init__(self, vcf_idx, allele, qscore=None):
        self.vcf_idx = vcf_idx
        self.allele = allele  # 0 or 1
        self.qscore = qscore

    def __str__(self):
        return "Variant: vcf_idx={} allele={} qscore={}".format(self.vcf_idx, self.allele, self.qscore)


class Block:
    def __init__(self, vcf_idx, alleles, qscores=None):
        self.vcf_idx_start = vcf_idx
        self.vcf_idx_end = vcf_idx + len(alleles) - 1
        self.variants = []
        for i in range(len(alleles)):
            qscore = None if (qscores is None) else qscores[i]
            self.variants.append(VariantHandle(vcf_idx + i, alleles[i], qscore))
        self.n_variants = len(self.variants)

    def __str__(self):
        return "Block: n_variants={} vcf_idx={} variants=\n".format(self.n_variants, self.vcf_idx_start) \
               + '\n'.join(map(str, self.variants))

    def overlaps(self, block):
        return (min(self.vcf_idx_end, block.vcf_idx_end) - max(self.vcf_idx_start, block.vcf_idx_start)) >= 0


class Fragment:
    def __init__(self, blocks=None):
        if blocks is None:
            blocks = []
        self.blocks = blocks
        self.n_blocks = len(blocks)
        self.n_variants = 0
        if len(self.blocks) > 0:
            for block in self.blocks:
                self.n_variants += block.n_variants

        self.read_id = None
        self.paired = None
        self.insert_size = None
        self.strand = None
        self.matepos = None
        self.read_barcode = None

    def overlap(self, fragment):
        shared_variants = []
        for b1 in self.blocks:
            for b2 in fragment.blocks:
                if b1.overlaps(b2):
                    # find the shared variants
                    for v1 in b1.variants:
                        for v2 in b2.variants:
                            if v1.vcf_idx == v2.vcf_idx:
                                shared_variants.append((v1, v2))
        return shared_variants

    @staticmethod
    def parse_from_file(frag_line):
        fields = frag_line.split()
        frag = Fragment()
        frag.n_blocks = int(fields[0])
        frag.read_id = fields[1]
        field_idx = 2
        # parse each block
        for _ in range(frag.n_blocks):
            vcf_index = int(fields[field_idx]) - 1  # -1 since the variant index is 1-based in the HapCut file
            alleles = fields[field_idx + 1]
            field_idx += 2
            frag.blocks.append(Block(vcf_index, alleles))
            frag.n_variants += len(alleles)
        # parse quality scores for all variants
        qscores = fields[field_idx]
        qscore_idx = 0
        for block in frag.blocks:
            for variant in block.variants:
                variant.qscore = qscores[qscore_idx]
                qscore_idx += 1

        return frag

    def __str__(self):
        return "\nread_id={} n_blocks={} blocks=\n".format(self.read_id, self.n_blocks) + '\n'.join(map(str, self.blocks))


def parse_frag_file(frags_fname):
    fragments = []
    for frag_line in open(frags_fname, 'r'):
        fragments.append(Fragment.parse_from_file(frag_line))
    return fragments


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load fragment file")
    parser.add_argument('--frags', help='Input fragment file')
    args = parser.parse_args()
    fragments = parse_frag_file(args.frags)

