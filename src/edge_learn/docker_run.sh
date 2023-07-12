function calculate_port() {
    local rank=$1
    local offset=9001
    echo $((2 * rank + offset))
}

# Build docker image
docker build -t edge_learn:latest ../..

machine_id=$(jq -r '.machine_id' config/params.json)
cloud_machine_id=$(jq -r '.cloud_machine_id' config/params.json)
clients_per_machine=$(jq -r '.clients_per_machine' config/params.json)

base_result_dir=$(pwd)/results/$(date +%Y-%m-%d_%H-%M)
mkdir -p $base_result_dir

# Run server if machine_id is cloud_machine_id
if [ $machine_id -eq $cloud_machine_id ]; then
    mkdir -p $base_result_dir/primary_cloud
    echo "Running primary cloud"
    docker run -d -p $(calculate_port -1):1000 -v $base_result_dir/primary_cloud:/results --name primary_cloud edge_learn:latest python3 create_node.py --node_type cloud --rank -1 --config_dir config 
fi 

# Run clients
for i in $(seq 0 $(($clients_per_machine - 1))); do
    mkdir -p $base_result_dir/client_$i
    echo "Running client $i"
    docker run -d -p $(calculate_port $((i + 1))):1000 -v $base_result_dir/client_$i:/results --name client_$i edge_learn:latest python3 create_node.py --node_type client --rank $((i + 1)) --config_dir config
done

# Run edge node
mkdir -p $base_result_dir/edge_server
echo "Running edge server"
docker run -it -p $(calculate_port 0):1000 -v $base_result_dir/edge_server:/results --name edge_server edge_learn:latest python3 create_node.py --node_type edge --rank 0 --config_dir config