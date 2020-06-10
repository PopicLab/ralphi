workflow Frags {
    # for each sample and chromosome:
    # -- extract VCF (single sample and single chromosome)
    # -- extract fragments
    String docker = "gcr.io/deepphase/seqtools:1.0"
    File input_vcf_dir
    File input_samples_panel
    Array[String] chromosomes
    Array[Array[String]] samples = read_tsv(input_samples_panel)

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

    Array[Pair[Array[String], String]] subsets = cross(samples, chromosomes)
    scatter (subset in subsets) {
        String sample = subset.left[0]
        String sample_bam = subset.left[1]
        String chr = subset.right
        call extract_vcf {
            input:
                input_vcf = "ALL.chr" + chr + ".fixed.vcf.gz",
                input_vcf_index = "ALL.chr" + chr + ".fixed.vcf.gz.tbi",
                output_vcf_fname = sample + ".chr" + chr + ".vcf",
                sample = sample,
                chr = chr,
                docker = docker,
                link = fix_vcf_header.output_vcf,
                link_index = fix_vcf_header.output_vcf_index
        }
        call extract_frags {
            input:
                input_vcf = extract_vcf.output_vcf,
                input_bam = sample_bam,
                output_frags_fname = basename(extract_vcf.output_vcf, "vcf") + "frags",
                docker = docker
        }
    }

    output {
        Array[File] output_frags = extract_frags.output_frags
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

task extract_vcf {
    File input_vcf
    File input_vcf_index
    String output_vcf_fname
    String sample
    String chr
    String docker
    Array[File] link
    Array[File] link_index

    command {
        bcftools view --types snps -I -a -ghet -c1 -Ov -s ${sample} --regions ${chr} -o ${output_vcf_fname} ${input_vcf}
    }

    output {
        File output_vcf = "${output_vcf_fname}"
    }

    runtime {
        docker: docker
        memory: "4 GiB"
        cpu: "1"
    }
}

task extract_frags {
    File input_vcf
    File input_bam
    String output_frags_fname
    String docker

    command {
        extractHAIRS --bam ${input_bam} --VCF ${input_vcf} --out ${output_frags_fname}
    }

    output {
        File output_frags = "${output_frags_fname}"
    }

    runtime {
        docker: docker
        memory: "4 GiB"
        cpu: "1"
    }
}