import vcf.filters
import vcf
from vcf.parser import Reader, Writer, _Format
import collections


def write_phased_vcf(input_vcf, idx2var, output_vcf, chromosome):
    vcf_reader = vcf.Reader(open(input_vcf, 'rb'), strict_whitespace=True)
    assert (len(vcf_reader.samples) == 1), "Only single-sample files are expected"
    input_format_keys = list(vcf_reader.formats.keys())
    # add phasing related format fields
    vcf_reader.formats['PS'] = _Format('PS', 1, 'Integer', 'ID of Phase Set for Variant')
    vcf_reader.formats['PD'] = _Format('PD', 1, 'Integer', 'Phased Read Depth')
    vcf_reader.formats['PQ'] = _Format('PQ', 1, 'Integer', 'Phred QV indicating probability that this variant is '
                                                           'incorrectly phased relative to the haplotype')
    writer = Writer(open(output_vcf, 'w'), vcf_reader)
    vcf_idx = 0
    processed_chr = False
    for record in vcf_reader:
        if record.CHROM != chromosome:
            if processed_chr: break
            continue
        processed_chr = True
        # update the genotype field
        mapping = {input_format_keys[i]: record.samples[0].data[i] for i in range(len(input_format_keys))}
        gt = record.genotype(record.samples[0].sample).data.GT
        if vcf_idx in idx2var and idx2var[vcf_idx].h != None:  # phased
            v = idx2var[vcf_idx]
            if v.h == 0:
                mapping['GT'] = str(0) + "|" + str(1)
            else:
                mapping['GT'] = str(1) + "|" + str(0)
            mapping['PS'] = str(v.phase_set)
            mapping['PD'] = str(len(v.frag_variants))
            mapping['PQ'] = '0'

        else:  # unphased
            mapping['GT'] = str(gt)
            mapping['PS'] = "."
            mapping['PD'] = "."
            mapping['PQ'] = '0'
        record.samples[0].data = collections.namedtuple('CallData', vcf_reader.formats.keys())._make(mapping[x]
                                                                            for x in vcf_reader.formats.keys())
        record.add_format('PS')
        record.add_format('PD')
        record.add_format('PQ')
        writer.write_record(record)
        vcf_idx += 1
    writer.close()
