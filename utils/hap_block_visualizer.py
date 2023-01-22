from tabulate import tabulate
from vcf.parser import Reader, Writer, _Format
import vcf
from termcolor import colored

def pretty_print_compressed(solutions, vars, input_vcf):
    solutions[0] = sorted(solutions[0], key=lambda frag: frag.vcf_idx_start)
    table = []
    raw_vcf_pos = []
    ground_truth_genotype = []
    raw_vcf_pos.append("")
    raw_vcf_pos.append("")
    raw_vcf_pos.append("Variant Position")
    ground_truth_genotype.append("Ground Truth")
    ground_truth_genotype.append("")
    ground_truth_genotype.append("")
    vcf_reader = vcf.Reader(open(input_vcf, 'r'), strict_whitespace=True)
    for record in vcf_reader:
        raw_vcf_pos.append(str(record.POS))
        ground_truth_genotype.append(str(record.genotype("HG00113").data))
    first_line = []
    first_line.append("Fragment #")
    first_line.append("Count")
    first_line.append("HAL Fragment Phasing")
    last_line = []
    last_line.append("HAL Phased Variants")
    last_line.append("-")
    last_line.append("-")
    for var in vars:
        first_line.append(str(var[1].vcf_idx))
        last_line.append(str(var[1].h))
    table.append(raw_vcf_pos)
    table.append(first_line)
    for i, frag in enumerate(solutions[0]):
        intermediate_line = []
        intermediate_line.append(str(i))
        intermediate_line.append(str(frag.n_copies))
        color = "blue"
        if frag.haplotype == 1.0:
            color = "red"
        intermediate_line.append(frag.haplotype)
        for i in range(len(vars)):
            intermediate_line.append("-")
        for block in frag.blocks:
            for var in block.variants:
                index = first_line.index(str(var.vcf_idx))
                intermediate_line[index] = colored(str(var.allele) + " (" + str(var.qscore) + ")", color)
        table.append(intermediate_line)
    phased_sets = []
    phased_sets.append("Phased Sets")
    phased_sets.append("-")
    phased_sets.append("-")
    for var in vars:
        phased_sets.append(str(var[1].phase_set))

    hap0 = []
    hap1 = []
    hap0.append("Hap0 Evidence")
    hap0.append("-")
    hap0.append("-")
    hap1.append("Hap1 Evidence")
    hap1.append("-")
    hap1.append("-")

    vars_isolated = [x[1] for x in vars]
    for var_item in vars_isolated:
        hap0_evidence = 0
        hap1_evidence = 0
        for v, n_copies in var_item.frag_variants:
            if (v.haplotype == 0 and v.allele == '0') or (v.haplotype == 1 and v.allele == '1'):
                hap0_evidence += n_copies
            elif (v.haplotype == 0 and v.allele == '1') or (v.haplotype == 1 and v.allele == '0'):
                hap1_evidence += n_copies
        hap0.append(str(hap0_evidence))
        hap1.append(str(hap1_evidence))

    table.append(phased_sets)
    table.append(hap0)
    table.append(hap1)
    table.append(last_line)
    table.append(ground_truth_genotype)
    return tabulate(table, headers='firstrow', tablefmt='fancy_grid')

def pretty_print(solutions, vars, input_vcf):
    solutions[0] = sorted(solutions[0], key=lambda frag: frag.vcf_idx_start)
    table = []
    raw_vcf_pos = []
    ground_truth_genotype = []
    raw_vcf_pos.append("")
    raw_vcf_pos.append("Variant Position")
    ground_truth_genotype.append("Ground Truth")
    ground_truth_genotype.append("")
    vcf_reader = vcf.Reader(open(input_vcf, 'r'), strict_whitespace=True)
    for record in vcf_reader:
        raw_vcf_pos.append(str(record.POS))
        ground_truth_genotype.append(str(record.genotype("HG00113").data))
    first_line = []
    first_line.append("Fragment #")
    first_line.append("HAL Fragment Phasing")
    last_line = []
    last_line.append("HAL Phased Variants")
    last_line.append("-")
    for var in vars:
        first_line.append(str(var[1].vcf_idx))
        last_line.append(str(var[1].h))
    table.append(raw_vcf_pos)
    table.append(first_line)
    for i, frag in enumerate(solutions[0]):
        intermediate_line = []
        intermediate_line.append(str(i))
        intermediate_line.append(frag.haplotype)
        color = "blue"
        if frag.haplotype == 1.0:
            color = "red"
        for i in range(len(vars)):
            intermediate_line.append("-")
        for block in frag.blocks:
            for var in block.variants:
                index = first_line.index(str(var.vcf_idx))
                intermediate_line[index] = colored(str(var.allele) + " (" + str(var.qscore) + ")", color)
        table.append(intermediate_line)
    phased_sets = []
    phased_sets.append("Phased Sets")
    phased_sets.append("-")
    for var in vars:
        phased_sets.append(str(var[1].phase_set))


    hap0 = []
    hap1 = []
    hap0.append("Hap0 Evidence")
    hap0.append("-")
    hap1.append("Hap1 Evidence")
    hap1.append("-")

    vars_isolated = [x[1] for x in vars]
    for var_item in vars_isolated:
        hap0_evidence = 0
        hap1_evidence = 0
        for v, n_copies in var_item.frag_variants:
            if (v.haplotype == 0 and v.allele == '0') or (v.haplotype == 1 and v.allele == '1'):
                hap0_evidence += n_copies
            elif (v.haplotype == 0 and v.allele == '1') or (v.haplotype == 1 and v.allele == '0'):
                hap1_evidence += n_copies
        hap0.append(str(hap0_evidence))
        hap1.append(str(hap1_evidence))

    table.append(phased_sets)
    table.append(hap0)
    table.append(hap1)
    table.append(last_line)
    table.append(ground_truth_genotype)
    return tabulate(table, headers='firstrow', tablefmt='fancy_grid')
