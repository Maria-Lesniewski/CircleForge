source ../../.venv/bin/activate

root=$(realpath ../../artifacts)

if [ ! -d ""$root"/feature_engineering_experiment" ]; then
	echo "This experiment uses the same training data as part2:feature_engineering"
	echo "Please complete that tutorial (or at least generate its data) before continuing here"
	exit 1
fi

python run_pipeline2.py
python run_pipeline3.py
