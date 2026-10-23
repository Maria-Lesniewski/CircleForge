#!/usr/bin/env bash

root=$(realpath ../../artifacts/intro_experiment)

cd $root

if [[ -d analysis_summary ]]; then
	echo "This script naively appends to files in the analysis_summary directory"
	echo "Please clean up/backup the directory before running this again"
	exit 1
fi

mkdir analysis_summary
cd analysis_summary

sets=("set_a" "set_b" "set_c" "set_d")
splits=("train" "val" "test")
grace_analyses=("r2" "rmsd")
## Stuff is wrapped like 
# set_X/ModelEvaluation/set_X'/BasicLinear/<split>/<analysis>

# <analysis for accuracy is a json file>
# <analysis for R2 and RMSD are in col 2 of grace/evaluation_X>

for split in "${splits[@]}"; do
	# TWO GRACE STYLE ANALYSES
	for analysis in "${grace_analyses[@]}"; do
		row_filename="$analysis".xvg
		echo "${split^^} ${split^^} ${split^^}" >> $row_filename
		printf '%24s\t' "MODEL / DATA set_a |" >> $row_filename
		printf '%24s\t' "set_b " >> $row_filename
		printf '%24s\t' "set_c " >> $row_filename
		printf '%24s\t\n' "set_d " >> $row_filename
		for data1 in "${sets[@]}";do
			data1_array=() # aggregate row-wise
			for data2 in "${sets[@]}";do
				analysis_file=$root/"$data1"/ModelEvaluation/"$data2"/BasicLinear/"$split"/"$analysis"/grace/evaluation_"${analysis^^}".xvg
#				echo "$analysis_file"
				#value=$(awk 'NR==1 {print $2; exit}' "$analysis_file")
				value=$(awk 'NR==1 {printf "%.10f\n", $2; exit}' "$analysis_file")
#				echo $value
				data1_array+=(""$value" ")
			done
			
			printf '%s\t' "$data1" >> $row_filename
			printf '%s\t' "${data1_array[@]}" >> $row_filename
			printf '\n' >> $row_filename
			echo "" >> $row_filename
		done
	done
	### NOW THE ACCURACY JSON
	row_filename="accuracy_with_tol.xvg"
	echo "${split^^} ${split^^} ${split^^}" >> $row_filename
        printf '%14s\t' " MODEL \DATA set_a |" >> $row_filename
        printf '%14s\t' "set_b |" >> $row_filename
        printf '%14s\t' "set_c |" >> $row_filename
        printf '%14s\t\n' "set_d |" >> $row_filename

	for data1 in "${sets[@]}";do
		data1_array=() # aggregate row-wise
			for data2 in "${sets[@]}";do
				analysis_file=$root/"$data1"/ModelEvaluation/"$data2"/BasicLinear/"$split"/accuracy_within_tol/accuracy.json
				value=$(awk 'NR==5 {printf "%.10f\n", $2; exit}' "$analysis_file")
				#value=$(awk 'NR==5 {print $2; exit}' $analysis_file)
	#			echo $value
				data1_array+=(""$value"")
			done
		printf '%s\t' "$data1" >> $row_filename
                printf '%s\t' "${data1_array[@]}" >> $row_filename
                printf '\n' >> $row_filename

	done
done
