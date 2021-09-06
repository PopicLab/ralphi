task hapcut2_phase {
    File input_vcf
    File input_frags
    command {
        ./HAPCUT2 --fragments ${input_frags} --VCF ${input_vcf} --output ${input_vcf}.phased.vcf
    }

    output {
        File output_frags = "${input_vcf}.phased.vcf"
    }
}

task phasing_stats {
    File phased_vcf
    File ground_truth_vcf

    command {
        python3 utilities/calculate_haplotype_statistics.py -v1 ${phased_vcf} -v2 ${ground_truth_vcf} > ${phased_vcf}.stats
    }

    output {
        File output_stats = "${phased_vcf}.stats"
    }
}