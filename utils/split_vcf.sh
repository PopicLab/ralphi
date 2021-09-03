
INPUT_VCF=$1
for sample in $(bcftools query -l "${INPUT_VCF}"); do
   bcftools view -c1 -Oz -s "$sample" -o "$sample".vcf.gz "${INPUT_VCF}"
done