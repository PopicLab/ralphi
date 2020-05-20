import argparse


class Fragment:
    def __init__(self):
        self.n_blocks = None
        self.read_id = None
        self.read_barcode = None
        self.n_variants = None
        self.insert_size = None
        self.strand = None

    @staticmethod
    def parse_hapcut_frag(frag_line):
        # blocks, id,
        fields = frag_line.split()
        frag = Fragment()
        frag.n_blocks = fields[0]
        frag.chr_id = fields[1]
        print(frag.to_string())

    def to_string(self):
        s = "{} {}".format(self.n_blocks, self.read_id)
        return s


def parse_frag_file(frags_fname):
    """
    """
    fragments = []
    for frag_line in open(frags_fname, 'r'):
        fragments.append(Fragment.parse_hapcut_frag(frag_line))
    return fragments


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load HapCut2 fragment file")
    parser.add_argument('--frags', help='Input fragment file')
    args = parser.parse_args()
    parse_frag_file(args.frags)

