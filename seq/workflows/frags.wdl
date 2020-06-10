workflow Frags {
    # for each sample and chromosome:
    # -- extract VCF (single sample and single chromosome)
    # -- extract fragments
    String docker = "gcr.io/deepphase/seqtools:1.0"
    File input_vcf_panel
    File input_samples_panel
    Array[String] chromosomes
    Array[Array[String]] vcf_per_chr = read_tsv(input_vcf_panel)
    Array[Array[String]] samples = read_tsv(input_samples_panel)

    Array[Pair[Array[String], Pair[String, Array[String]]]] subsets = cross(samples, zip(chromosomes, vcf_per_chr))
    scatter (subset in subsets) {
        String sample = subset.left[0]
        String sample_bam = subset.left[1]
        String chr = subset.right.left
        String chr_vcf = subset.right.right[0]
        call extract_vcf {
            input:
                input_vcf = chr_vcf,
                input_vcf_index = chr_vcf + ".tbi",
                output_vcf_fname = sample + ".chr" + chr + ".vcf",
                sample = sample,
                chr = chr,
                docker = docker,
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

task extract_vcf {
    File input_vcf
    File input_vcf_index
    String output_vcf_fname
    String sample
    String chr
    String docker

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
        disks: "local-disk 32 SSD"
    }
}