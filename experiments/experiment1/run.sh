echo "Grabbing python environment.."

source ../../.venv/bin/activate

for ele in 0 1 2
do
	echo "$ele" | python direct_linear_model_solver.py
done


for i in 1 2 3 4 5 6 7 8 9; 
do
	echo "0 4 5 -1 ";
	echo "1";
done| python analyze_models.py

