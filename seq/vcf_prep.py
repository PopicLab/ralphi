import vcf.filters
import vcf
import argparse
from vcf.parser import Reader, Writer


class HetSNPOnly(vcf.filters.Base):
    """Keep only heterozygous SNPs"""

    name = 'het-snp-only'

    def __call__(self, record):
        if not record.is_snp or any([not record.genotype(sample.sample).is_het for sample in record.samples]):
            return True


def split_by_chr(vcf_fname, filters):
    """
    Split the input VCF file by chromosome, apply filters
    """
    chroms = ['chr{}'.format(x) for x in range(1, 23)] + ['chrX', 'chrY']
    output_fnames = [vcf_fname + '.{}.vcf'.format(c) for c in chroms]
    vcf_reader = vcf.Reader(open(vcf_fname, 'r'), prepend_chr=True, strict_whitespace=True)
    assert(len(vcf_reader.samples) == 1), "Only single-sample files are expected"
    writers = {chrom: Writer(open(f, 'w'), vcf_reader) for (chrom, f) in zip(chroms, output_fnames)}
    for record in vcf_reader:
        keep = all([f(record) is not None for f in filters])
        if keep:
            writers[record.CHROM].write_record(record)


def filter_vcf(vcf_fname, filters):
    vcf_reader = vcf.Reader(open(vcf_fname, 'r'), prepend_chr=True, strict_whitespace=True)
    assert (len(vcf_reader.samples) == 1), "Only single-sample files are expected"
    output_fname = vcf_fname + '.filtered.vcf'
    writer = Writer(open(output_fname, 'w'), vcf_reader)
    for record in vcf_reader:
        keep = all([f(record) is not None for f in filters])
        if keep:
            writer.write_record(record)


def extract_vcf_for_specific_variants(vcf_positions, input_vcf, output_vcf):
    print("constructing vcf for specific graph --", " input vcf: ", input_vcf, " output vcf: ", output_vcf)
    vcf_reader = vcf.Reader(open(input_vcf, 'r'), strict_whitespace=True)
    assert (len(vcf_reader.samples) == 1), "Only single-sample files are expected"

    writer = Writer(open(output_vcf, 'w'), vcf_reader)
    max_vcf_idx = max(vcf_positions)
    for vcf_idx, record in enumerate(vcf_reader):
        if vcf_idx > max_vcf_idx:
            break
        if vcf_idx not in vcf_positions:
            continue
        writer.write_record(record)
    writer.close()
    print("wrote vcf file to:", output_vcf)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare input VCF")
    parser.add_argument('--vcf', help='Input VCF')
    parser.add_argument('--chr', default=False, help='Split by chromosome')
    args = parser.parse_args()
    if args.chr:
        split_by_chr(args.vcf, [HetSNPOnly(None)])
    else:
        filter_vcf(args.vcf, [HetSNPOnly(None)])

