cd /scratch/stervan/llm-trees/
conda activate conda-env/
#nohup /scratch/stervan/llm-trees/conda-env/bin/python -m ecml_calculations > output.log 2>&1 < /dev/null &
#nohup /scratch/stervan/llm-trees/conda-env/bin/python ecml_calculations > output.log 2>&1 &
nohup python3 -m ecml_calculations > output.log 2>&1 &
#tail -f output.log | awk -e '{ print strftime("%Y%m%d_%H%M%S",systime()) "\t" $0}'
#jobs -l
pgrep -af ecml_calculations
#kill 12345 # kill a running job by id