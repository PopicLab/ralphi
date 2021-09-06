INPUT_PANEL=$1

while IFS="" read -r file || [ -n "$file" ]
do
  gsutil copy "${file}" .
done < "${INPUT_PANEL}"