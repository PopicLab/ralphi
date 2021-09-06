workflow FixVCF {
    String docker = "gcr.io/deepphase/seqtools:1.0"
    File input_vcf_dir
    File input_samples_panel
    Array[String] chromosomes

    scatter (chr in chromosomes) {
        String chr_vcf = input_vcf_dir + "ALL.chr" + chr + ".phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf.gz"
        call fix_vcf_header {
            input:
                input_vcf = chr_vcf,
                input_vcf_index = chr_vcf + ".tbi",
                output_vcf_fname = "ALL.chr" + chr + ".fixed.vcf.gz",
                docker = docker
        }
    }

    output {
        Array[File] vcfs = fix_vcf_header.output_vcf
        Array[File] vcf_idxs = fix_vcf_header.output_vcf_index
    }

}

task fix_vcf_header {
    File input_vcf
    File input_vcf_index
    String output_vcf_fname
    String docker

    command {
        echo "##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">" > header.txt
        bcftools annotate -O z --header-lines header.txt ${input_vcf} -o ${output_vcf_fname}
        bcftools index --tbi ${output_vcf_fname}
    }

    output {
        File output_vcf = "${output_vcf_fname}"
        File output_vcf_index = "${output_vcf_fname}.tbi"
    }

    runtime {
        docker: docker
        memory: "4 GiB"
        cpu: "1"
    }
}