import engine.config as config_utils

import argparse
from collections import defaultdict
from copy import deepcopy
import logging
import sys
import whatshap
from whatshap.cli import PhasedInputReader

# Fragment generation utility, includes:
# -- realignment and read selection adapted from Whatshap
# -- custom variant filtering

class NestedDict(defaultdict):
    def __call__(self):
        return NestedDict(self.default_factory)

def generate_fragments(config):
    bam_reader = PhasedInputReader(config.bams, config.reference, None, ignore_read_groups=True,
                                   mapq_threshold=config.mapq if config.platform == "illumina" else 0,
                                   only_snvs=True, overhang=config.realign_overhang)
    vcf_reader = whatshap.vcf.VcfReader(config.vcf, only_snvs=True)
    assert len(vcf_reader.samples) == 1
    variant_table = next(vcf_reader.__iter__())
    logging.info("Processing %s", variant_table.chromosome)
    variant_table = select_phaseable_variants(vcf_reader.samples[0], variant_table)
    logging.info("Number of variants loaded: %d", len(variant_table))
    reads, _ = bam_reader.read(variant_table.chromosome, variant_table.variants, None, read_vcf=False, args=config)
    logging.info("Number of reads loaded: %d", len(reads))
    if config.log_reads:
        write_reads(reads, config.out_dir + "/reads_all.txt")
    reads, n_filtered = select_variants(reads, config)
    logging.info("Number of variants filtered: %d", n_filtered)
    # keep only reads that cover at least two variants and have a sufficiently high MAPQ
    reads = reads.subset([i for i, read in enumerate(reads) if len(read) >= 2 and read.mapqs[0] >= config.mapq])
    if config.enable_read_selection:
        logging.debug("Running read selection to a maximum coverage of: %dX", config.max_coverage)
        reads = reads.subset(whatshap.readselect.readselection(reads, config.max_coverage))
    if config.log_reads:
        write_reads(reads, config.out_dir + "/reads.txt")
    logging.info("[FINAL] Kept %d reads covering %d variants", len(reads), len(reads.get_positions()))
    write_fragments(reads, variant_table.variants, config.out_dir + "/fragments.txt")

def write_fragments(reads, variants, out_file_name):
    pos2idx = {}
    for v in variants:
        pos2idx[v.position] = v.index
    frags_file = open(out_file_name, "w")
    for read in reads:
        blocks = get_fragment_blocks(read, pos2idx)
        block_allele_str = " ".join(str(b[0]) + " " + b[1] for b in blocks)
        block_qual_str = "".join(str(b[2]) for b in blocks)
        assert len(blocks) > 0
        assert len(block_qual_str) > 1
        print(len(blocks), read.name, block_allele_str, block_qual_str, sep=" ", file=frags_file)


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


def write_reads(reads, output_fname):
    reads_file = open(output_fname, "w")
    for read in reads:
        if len(read):
            print(read.name, read.strand, len(read), read.mapqs[0],
                  read[0].position + 1, read[-1].position + 1,
                  ",".join([str(read[i].position + 1) for i in range(len(read))]),
                  ",".join([str(read[i].allele) for i in range(len(read))]),
                  ",".join([str(read[i].quality) for i in range(len(read))]),
                  sep="\t", file=reads_file)
        else:
            print(read.name, read.strand, len(read), read.mapqs[0],
                  sep="\t", file=reads_file)


def select_phaseable_variants(sample, variant_table):
    variants_to_filter = set()
    for index, gt in enumerate(variant_table.genotypes_of(sample)):
        if gt.is_none() or gt.is_homozygous():
            variants_to_filter.add(index)
    variants_to_phase = deepcopy(variant_table)
    variants_to_phase.remove_rows_by_index(variants_to_filter)
    logging.info("Found %d usable heterozygous variants (%d filtered out)",
                 len(variants_to_phase), len(variants_to_filter))
    return variants_to_phase


def select_variants(reads, config):
    pos2alleles = NestedDict(NestedDict(NestedDict(int)))
    pos2coverage = NestedDict(NestedDict(int))
    pos2strands = defaultdict(set)
    # 1. collect allele/coverage information for each variant site
    for read in reads:
        for variant in read:
            if config.mbq and variant.quality < config.mbq: continue
            pos2coverage['all'][variant.position] += 1
            pos2alleles['all'][variant.position][variant.allele] += 1
            if read.mapqs[0] == 0:
                pos2coverage['mapq0'][variant.position] += 1
                pos2alleles['mapq0'][variant.position][variant.allele] += 1
            else:
                pos2coverage['mapq1'][variant.position] += 1
                pos2alleles['mapq1'][variant.position][variant.allele] += 1
                if read.mapqs[0] >= config.mapq:
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
            continue

        # no reads with high mapq cover this variant
        if not pos2coverage['high'][variant_position]:
            filter_by_type['high_cov'] += 1
            filtered_variants.add(variant_position + 1)
            continue

        # the fraction of high mapq reads covering this variant is too low
        if pos2coverage['high'][variant_position] / pos2coverage['all'][variant_position] < config.min_highmapq_ratio:
            filter_by_type['high_cov_ratio'] += 1
            filtered_variants.add(variant_position + 1)
            continue

        # the fraction of mapq1 reads covering this variant is too low
        if pos2coverage['mapq1'][variant_position] / pos2coverage['all'][variant_position] < config.min_mapq1_ratio:
            filter_by_type['mapq1_ratio'] += 1
            filtered_variants.add(variant_position + 1)
            continue

        # the fraction of reads with a bad allele
        bad_allele_ratio = pos2alleles['mapq1'][variant_position][-1] / pos2coverage['mapq1'][variant_position]
        if bad_allele_ratio >= config.max_bad_allele_ratio:
            filter_by_type['bad_allele_ratio'] += 1
            filtered_variants.add(variant_position + 1)
            continue

        # if we have the minimum required coverage to filter
        if config.enable_strand_filter and pos2coverage['all'][variant_position] >= config.min_coverage_to_filter_ref:
            # filter variants covered by only one strand
            if len(pos2strands[variant_position]) == 1:
                filter_by_type['strand'] += 1
                filtered_variants.add(variant_position + 1)
                continue

            # filter variants with a low fraction of ALT alleles
            if 1 in pos2alleles['mapq1'][variant_position]:
                alt_allele_ratio = pos2alleles['mapq1'][variant_position][1] / pos2coverage['mapq1'][variant_position]
                if alt_allele_ratio < config.min_alt_allele_ratio:
                    filter_by_type['alt_allele_ratio'] += 1
                    filtered_variants.add(variant_position + 1)
                    continue

        # filter variants covered only by a single allele
        if len(pos2alleles['mapq1'][variant_position]) == 1:
            (allele,) = pos2alleles['mapq1'][variant_position]
            if allele == -1:  # only bad alleles cover the variant, always filter
                filter_by_type['single_allele_bad'] += 1
                filtered_variants.add(variant_position + 1)
                continue
            elif allele == 0 and pos2coverage['all'][variant_position] >= config.min_coverage_to_filter_ref:
                filter_by_type['single_allele_ref'] += 1
                filtered_variants.add(variant_position + 1)
                continue
            elif allele == 1 and pos2coverage['all'][variant_position] >= config.min_coverage_to_filter_alt:
                filter_by_type['single_allele_alt'] += 1
                filtered_variants.add(variant_position + 1)
                continue
        else:  # we have at least two different alleles
            if len(pos2alleles['mapq1'][variant_position]) == 2 and -1 in pos2alleles['mapq1'][variant_position]:
                # one of the alleles is a bad allele, the others are either all ALT or REF
                if 0 in pos2alleles['mapq1'][variant_position] and pos2coverage['all'][variant_position] >= \
                       config.min_coverage_to_filter_ref:
                    filter_by_type['double_allele_with_ref_and_bad'] += 1
                    filtered_variants.add(variant_position + 1)
                    continue
                elif 1 in pos2alleles['mapq1'][variant_position] and pos2coverage['all'][variant_position] >= \
                       config.min_coverage_to_filter_alt:
                    filter_by_type['double_allele_with_alt_and_bad'] += 1
                    filtered_variants.add(variant_position + 1)
                    continue

    print(filter_by_type)
    for read in reads:
        read_filtered_ids = []
        for i, variant in enumerate(read):
            if variant.quality < config.mbq:
                read_filtered_ids.append(variant.position)
                continue
            assert pos2coverage['all'][variant.position] >= 1
            if variant.allele == -1:
                read_filtered_ids.append(variant.position)
                continue
            if variant.position + 1 in filtered_variants:
                read_filtered_ids.append(variant.position)
        for p in read_filtered_ids:
            read.remove_variant(p)

    # print(sorted(filtered_variants))
    return reads, len(filtered_variants)


def main():
    print("*********************************")
    print("*  ralphi: fragment generation  *")  # TODO: add ralphi version
    print("*********************************")
    # ------ CLI ------
    parser = argparse.ArgumentParser(description='Fragment generation for phasing')
    parser.add_argument('--config', help='Fragment configuration YAML')
    args = parser.parse_args()
    # -----------------
    config = config_utils.load_config(args.config, config_type=config_utils.CONFIG_TYPE.FRAGS)
    generate_fragments(config)


main()