import vcf
from vcf.parser import Writer, _Format
import collections


def write_phased_vcf(input_vcf, idx2var, output_vcf, chromosome):
    vcf_reader = vcf.Reader(open(input_vcf, 'rb'), strict_whitespace=True)
    assert (len(vcf_reader.samples) == 1), "Only single-sample files are expected"

    # add phasing related format fields
    if 'PS' not in vcf_reader.formats:
        vcf_reader.formats['PS'] = _Format('PS', 1, 'Integer', 'ID of Phase Set for Variant')
    if 'PD' not in vcf_reader.formats:
        vcf_reader.formats['PD'] = _Format('PD', 1, 'Integer', 'Phased Read Depth')
    if 'PQ' not in vcf_reader.formats:
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
        record_format_keys = record.FORMAT.split(':')
        mapping = {record_format_keys[i]: record.samples[0].data[i] for i in range(len(record_format_keys))}
        if vcf_idx in idx2var and idx2var[vcf_idx].phase_set is not None:  # phased
            v = idx2var[vcf_idx]
            if v.h == 0:
                mapping['GT'] = str(0) + "|" + str(1)
            else:
                mapping['GT'] = str(1) + "|" + str(0)
            mapping['PS'] = str(v.phase_set)
            mapping['PD'] = str(len(v.frag_variants))
            mapping['PQ'] = '0'

        else:  # unphased
            mapping['GT'] = str(0) + "/" + str(1)
            mapping['PS'] = "."
            mapping['PD'] = "."
            mapping['PQ'] = '0'

        new_record_keys = record_format_keys.copy()
        for key in ['PS', 'PD', 'PQ']:
            if key not in new_record_keys:
                new_record_keys.append(key)

        record.samples[0].data = collections.namedtuple('CallData', new_record_keys)._make(
            mapping.get(x, ".") for x in new_record_keys
        )
        record.FORMAT = ':'.join(new_record_keys)
        writer.write_record(record)
        vcf_idx += 1
    writer.close()
