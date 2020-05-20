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


def prepare_vcf(vcf_fname):
    """
    Split the input VCF file by chromosome, keep only heterozygous SNPs
    """
    chroms = ['chr{}'.format(x) for x in range(1, 23)] + ['chrX', 'chrY']
    output_fnames = [vcf_fname + '.{}.vcf'.format(c) for c in chroms]
    vcf_reader = vcf.Reader(open(vcf_fname, 'r'), prepend_chr=True, strict_whitespace=True)
    assert(len(vcf_reader.samples) == 1), "Only single-sample files are expected"
    writers = {chrom: Writer(open(f, 'w'), vcf_reader) for (chrom, f) in zip(chroms, output_fnames)}
    filters = [HetSNPOnly(None)]
    for record in vcf_reader:
        for f in filters:
            if f(record) is None:
                writers[record.CHROM].write_record(record)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare input VCF")
    parser.add_argument('--vcf', help='Input VCF')
    args = parser.parse_args()
    prepare_vcf(args.vcf)
