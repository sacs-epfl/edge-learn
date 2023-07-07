#!/bin/bash

decpy_path=../eval # Path to eval folder
run_path=../eval/data # Path to the folder where the graph and config file will be copied and the results will be stored
config_file=config.ini
cp $config_file $run_path

env_python=/Users/lovelace/anaconda3/bin/python3 # Path to python executable of the environment | conda recommended
machines=1 # number of machines in the runtime
iterations=500
test_after=20
eval_file=$decpy_path/executeHierarchical.py
log_level=DEBUG # DEBUG | INFO | WARN | CRITICAL

server_machine=0 # machine id corresponding consistent with ip.json
m=0 # machine id corresponding consistent with ip.json
echo M is $m

procs_per_machine=3 
echo procs per machine is $procs_per_machine
batch_size=15
echo batch size is $batch_size

log_dir=$run_path/$(date '+%Y-%m-%dT%H:%M')/machine$m # in the eval folder
mkdir -p $log_dir

# export PYTHONPATH="${PYTHONPATH}:${PWD}/../../.."
# echo $PYTHONPATH

echo "Starting the execution"
$env_python $eval_file -ta $test_after -ld $log_dir -mid $m -ps $procs_per_machine -ms $machines -its $iterations -cf $run_path/$config_file -ll $log_level -sm $server_machine -bs $batch_size