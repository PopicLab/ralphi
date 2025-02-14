import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from intervaltree import IntervalTree
import logging
import whatshap
import string
from whatshap import readselect
from whatshap.cli import PhasedInputReader
from whatshap import merge

class FragVariantHandle:
    def __init__(self, vcf_idx, allele, qscore=None):
        self.vcf_idx = vcf_idx
        self.allele = allele
        self.qscore = qscore
        self.haplotype = None

    def __str__(self):
        return "Variant: vcf_idx={} allele={} qscore={}".format(self.vcf_idx, self.allele, self.qscore)

    def __eq__(self, v):
        return(self.vcf_idx, self.allele, self.haplotype) == (v.vcf_idx, v.allele, v.haplotype)


class Block:
    def __init__(self, vcf_idx, alleles, qscores=None):
        self.vcf_idx_start = vcf_idx
        self.vcf_idx_end = vcf_idx + len(alleles) - 1
        self.variants = []
        for i in range(len(alleles)):
            qscore = None if (qscores is None) else qscores[i]
            self.variants.append(FragVariantHandle(vcf_idx + i, alleles[i], qscore))
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
    def __init__(self, read_name, blocks=None, variants=None, n_copies=1):
        self.read_id = read_name
        self.n_copies = n_copies  # number of copies of this fragment
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
        self.fragment_group_id = None

    def __eq__(self, fragment):
        return (self.vcf_idx_start, self.vcf_idx_end, self.n_variants, self.variants) == (
            fragment.vcf_idx_start, fragment.vcf_idx_end, fragment.n_variants, fragment.variants)

    def __str__(self):
        return "\nread_id={} n_variants={} variants=\n".format(self.read_id, self.n_variants) + \
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
        field_idx = 1
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
                variant.qscore = convert_qscore(qscores[qscore_idx])
                qscore_idx += 1
        return Fragment(fields[1], blocks=blocks)

    def split_at(self, position):
        split_index = next(i for i, variant in enumerate(self.variants) if variant.vcf_idx >= position)
        return Fragment(self.read_id, variants=self.variants[:split_index], n_copies=self.n_copies),
               Fragment(self.read_id, variants=self.variants[split_index:], n_copies=self.n_copies)


def convert_qscore(qscore):
    return 10 ** ((ord(qscore) - 33) / (-10))


def parse_frag_file(frags_fname):
    fragments = []
    with open(frags_fname, 'r') as f:
        for frag_line in f:
            fragments.append(Fragment.parse_from_file(frag_line))
    # sort fragments to optimize graph construction
    fragments = sorted(fragments, key=lambda frag: frag.vcf_idx_start)
    return fragments

def parse_frag_repr(frag_repr):
    fragments = []
    for frag_line in frag_repr:
        fragments.append(Fragment.parse_from_file(frag_line))
    fragments = sorted(fragments, key=lambda frag: frag.vcf_idx_start)
    return fragments


def split_articulation(fragments):
    frag2split = defaultdict(list)
    frag2evidence = defaultdict(list)
    vcf_positions = set()
    fragment_intervals = IntervalTree()
    for i, frag in enumerate(fragments):
        fragment_intervals.addi(frag.vcf_idx_start, frag.vcf_idx_end+1, i)
        for var in frag.variants:
            vcf_positions.add(var.vcf_idx)
    vcf_positions = sorted(vcf_positions)
    for index, position in enumerate(vcf_positions[1:], start=1):
        incident_frag_idx = None
        for entry in sorted(fragment_intervals.at(position)):
            i = entry.data
            if fragments[i].vcf_idx_start < position:
                if not incident_frag_idx:
                    incident_frag_idx = i #entry.data
                else:
                    incident_frag_idx = None
                    break
        if incident_frag_idx:
            if vcf_positions[index - 1] not in frag2evidence[incident_frag_idx]:
                frag2split[incident_frag_idx].append(position)
            frag2evidence[incident_frag_idx].append(position)
    split_fragments = []
    for frag_index, frag in enumerate(fragments):
        if not frag2split[frag_index]:
            split_fragments.append(frag)
        else:  # iteratively split fragment at each splitting location
            cur_frag = frag
            for elem in sorted(frag2split[frag_index]):
                if cur_frag.vcf_idx_start < elem <= cur_frag.vcf_idx_end:
                    left, right = cur_frag.split_at(elem)
                    if left.vcf_idx_start:
                        left.fragment_group_id = frag_index
                        split_fragments.append(left)
                    if not right.vcf_idx_start: break
                    cur_frag = right
            if cur_frag.vcf_idx_start:
                cur_frag.fragment_group_id = frag_index
                split_fragments.append(cur_frag)
    return sorted(split_fragments, key=lambda frag: frag.vcf_idx_start)


################################
## Fragment generation utils  ##
################################

# -- realignment and read selection adapted from WhatsHap (Martin et al, 2016)
# -- basic variant filtering

class NestedDict(defaultdict):
    def __call__(self):
        return NestedDict(self.default_factory)


def generate_fragments(config, chromosome):
    bam_reader = PhasedInputReader([config.bam], config.reference, None, ignore_read_groups=True,
                                   mapq_threshold=config.mapq, only_snvs=True, overhang=config.realign_overhang)
    vcf_reader = whatshap.vcf.VcfReader(config.vcf, only_snvs=True)
    assert len(vcf_reader.samples) == 1
    variant_table = vcf_reader.fetch(chromosome)  # next(vcf_reader.__iter__())
    logging.debug("Processing %s", variant_table.chromosome)
    variant_table = select_phaseable_variants(vcf_reader.samples[0], variant_table)
    logging.debug("Number of variants loaded: %d", len(variant_table))
    reads, _ = bam_reader.read(variant_table.chromosome, variant_table.variants, None, read_vcf=False, args=config)
    logging.debug("Number of reads loaded: %d", len(reads))
    if config.log_reads: write_reads(reads, config.out_dir + "/input_reads.txt")
    if not config.no_filter:
        reads = select_variants(reads, config, variant_table)
    # keep only reads that cover at least two variants and have a sufficiently high MAPQ
    reads = reads.subset([i for i, read in enumerate(reads) if len(read) >= 2 and read.mapqs[0] >= config.mapq])
    logging.debug("Number of reads covering two variants : %d", len(reads))
    if config.enable_read_selection:
        logging.debug("Running read selection to a maximum coverage of: %dX", config.max_coverage)
        reads = reads.subset(readselect.readselection(reads, config.max_coverage))
    if config.log_reads: write_reads(reads, config.out_dir + "/output_reads.txt")
    logging.info("Selected %d reads covering %d variants in %s", len(reads), len(reads.get_positions()), chromosome)
    return assemble_fragments(reads, variant_table.variants)

def assemble_fragments(reads, variants):
    pos2idx = {}
    for v in variants:
        pos2idx[v.position] = v.index
    fragments = []
    for read in reads:
        blocks = get_fragment_blocks(read, pos2idx)
        block_allele_str = " ".join(str(b[0]) + " " + b[1] for b in blocks)
        block_qual_str = "".join(str(c) if c in string.printable else '!' for b in blocks for c in b[2])
        fragments.append(" ".join([str(len(blocks)), block_allele_str, block_qual_str]))
    return fragments


def get_fragment_blocks(read, pos2idx):
    prev_vcf_index = None
    current_block_alleles = ""
    current_block_quals = ""
    current_block_idx = None
    blocks = []
    for variant in read:
        assert variant.allele != -1
        vcf_index = pos2idx[variant.position] + 1
        if current_block_idx is None:
            current_block_idx = vcf_index
        if prev_vcf_index and vcf_index > prev_vcf_index + 1:
            blocks.append((current_block_idx, current_block_alleles, current_block_quals))
            current_block_alleles = str(variant.allele)
            current_block_quals = chr(variant.quality + 33)
            current_block_idx = vcf_index
        else:
            current_block_alleles += str(variant.allele)
            current_block_quals += chr(variant.quality + 33)
        prev_vcf_index = vcf_index
    if current_block_alleles:
        blocks.append((current_block_idx, current_block_alleles, current_block_quals))
    return blocks

def select_variants(reads, config, variant_table):
    pos2alleles = NestedDict(NestedDict(NestedDict(int)))
    pos2coverage = NestedDict(NestedDict(int))
    pos2strands = defaultdict(set)
    # 1. collect allele/coverage information for each variant site
    for read in reads:
        for variant in read:
            if config.mbq and variant.quality < config.mbq: continue
            pos2coverage['all'][variant.position] += 1
            pos2alleles['all'][variant.position][variant.allele] += 1
            if read.mapqs[0] == 60:
                pos2coverage['high'][variant.position] += 1
                pos2alleles['high'][variant.position][variant.allele] += 1
            pos2strands[variant.position].add(read.strand)

    # 2. apply filters at each variant site
    filtered_variants = set()
    filter_by_type = NestedDict(int)
    for variant_position in pos2coverage['all']:
        # too many reads cover this variant
        if pos2coverage['all'][variant_position] >= config.max_snp_coverage:
            filter_by_type['max_cov'] += 1
            filtered_variants.add(variant_position + 1)

        # no reads with high mapq cover this variant
        if config.require_hiqh_mapq and not pos2coverage['high'][variant_position]:
            filter_by_type['high_cov'] += 1
            filtered_variants.add(variant_position + 1)

            # the fraction of high mapq reads covering this variant is too low
            if pos2coverage['high'][variant_position] / pos2coverage['all'][variant_position] < config.min_highmapq_ratio:
                filter_by_type['high_cov_ratio'] += 1
                filtered_variants.add(variant_position + 1)

        if config.enable_strand_filter and pos2coverage['all'][variant_position] >= config.min_coverage_strand:
            # filter variants covered by only one strand
            if len(pos2strands[variant_position]) == 1:
                filter_by_type['strand'] += 1
                filtered_variants.add(variant_position + 1)

        # filter variants covered only by a single allele
        if len(pos2alleles['all'][variant_position]) == 1:
            (allele,) = pos2alleles['all'][variant_position]
            if allele == -1:  # only bad alleles cover the variant, always filter
                filter_by_type['single_allele_bad'] += 1
                filtered_variants.add(variant_position + 1)
            elif allele == 0 and pos2coverage['all'][variant_position] >= config.min_coverage_to_filter:
                filter_by_type['single_allele_ref'] += 1
                filtered_variants.add(variant_position + 1)
            elif allele == 1 and pos2coverage['all'][variant_position] >= config.min_coverage_to_filter:
                filter_by_type['single_allele_alt'] += 1
                filtered_variants.add(variant_position + 1)
        else:  # we have at least two different alleles
            if len(pos2alleles['all'][variant_position]) == 2 and -1 in pos2alleles['all'][variant_position]:
                # one of the alleles is a bad allele, the others are either all ALT or REF
                if 0 in pos2alleles['all'][variant_position] and pos2coverage['all'][variant_position] >= \
                        config.min_coverage_to_filter:
                    filter_by_type['double_allele_with_ref_and_bad'] += 1
                    filtered_variants.add(variant_position + 1)
                elif 1 in pos2alleles['all'][variant_position] and pos2coverage['all'][variant_position] >= \
                        config.min_coverage_to_filter:
                    filter_by_type['double_allele_with_alt_and_bad'] += 1
                    filtered_variants.add(variant_position + 1)


    logging.debug("Number of variants filtered: %d" % len(filtered_variants))
    logging.debug("Variants filtered by types: %s" %  filter_by_type)
    # remove variants from reads
    for read in reads:
        read_filtered_ids = set()
        for i, variant in enumerate(read):
            if variant.quality < config.mbq or variant.allele == -1:
                read_filtered_ids.add(variant.position)
            elif variant.position + 1 in filtered_variants:
                read_filtered_ids.add(variant.position)
        for p in read_filtered_ids:
            read.remove_variant(p)
    return reads

def select_phaseable_variants(sample, variant_table):
    variants_to_filter = set()
    for index, gt in enumerate(variant_table.genotypes_of(sample)):
        if gt.is_none() or gt.is_homozygous():
            variants_to_filter.add(index)
    variants_to_phase = deepcopy(variant_table)
    variants_to_phase.remove_rows_by_index(variants_to_filter)
    return variants_to_phase
