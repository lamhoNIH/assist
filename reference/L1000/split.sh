source="/Volumes/GoogleDrive/Shared drives/NIAAA_ASSIST/Data/LINCS L1000 (from GEO)/merged_L5_all_genes_repurposable_drugs.txt"

echo first part
date
head -3083 "${source}" > L1000-1.tsv 

# new_first_line.tsv was created before hand, but it could have been created using:
#1. head -1 L1000-1.tsv > origina_first_line.tsv
#2. cp origina_first_line.tsv new_first_line.tsv
#3. sed -i "" 's/\./#/g' new_first_line.tsv
#4. using vi to insert "Entrez id" and then tab at the very beginning

tail -n +2 L1000-1.tsv > out
cat new_first_line.tsv out > L1000-1.tsv
echo second part
date
sed -n '3084,6165p;6166q' "${source}" > out
cat new_first_line.tsv out > L1000-2.tsv
echo third part
date
sed -n '6166,9247p;9248q' "${source}" > out 
cat new_first_line.tsv out > L1000-3.tsv
echo fourth part
date
sed -n '9248,12329p;' "${source}" > out 
cat new_first_line.tsv out > L1000-4.tsv
date
rm out

