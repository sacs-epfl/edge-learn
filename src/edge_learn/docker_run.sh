calculate_port() {
    local rank=$1
    local offset=9001
    echo $((2 * rank + offset))
}

create_primary_cloud() {
    local base_result_dir=$1
    mkdir -p $base_result_dir/primary_cloud
    echo "Running primary cloud"
    docker run -d -p $(calculate_port -1):1000 -v $base_result_dir/primary_cloud:/results --name primary_cloud edge_learn:latest python3 create_node.py --node_type cloud --rank -1 --config_dir config 
}

is_primary_cloud() {
    local machine_id=$(jq -r '.machine_id' config/params.json)
    local cloud_machine_id=$(jq -r '.cloud_machine_id' config/params.json)
    if [ $machine_id -eq $cloud_machine_id ]; then
        echo 0
    else
        echo 1
    fi
}

create_primary_cloud_if_needed() {
    if [ $(is_primary_cloud) -eq 0 ]; then
        create_primary_cloud $base_result_dir
    fi 
}

create_edge_server() {
    local base_result_dir=$1
    mkdir -p $base_result_dir/edge_server
    echo "Running edge server"
    docker run -d -p $(calculate_port 0):1000 -v $base_result_dir/edge_server:/results --name edge_server edge_learn:latest python3 create_node.py --node_type edge --rank 0 --config_dir config
}

create_client() {
    local i=$1
    local base_result_dir=$2
    mkdir -p $base_result_dir/client_$i
    echo "Running client $i"
    docker run -d -p $(calculate_port $((i + 1))):1000 -v $base_result_dir/client_$i:/results --name client_$i edge_learn:latest python3 create_node.py --node_type client --rank $((i + 1)) --config_dir config
}

create_clients() {
    local base_result_dir=$1
    local clients_per_machine=$(jq -r '.clients_per_machine' config/params.json)
    for i in $(seq 0 $(($clients_per_machine - 1))); do
        create_client $i $base_result_dir
    done
}

launch_nodes() {
    base_result_dir=$(pwd)/results/$(date +%Y-%m-%d_%H-%M)
    mkdir -p $base_result_dir

    create_primary_cloud_if_needed $base_result_dir
    create_edge_server $base_result_dir
    create_clients $base_result_dir
}

wait_for_exit() {
    local container_name=$1
    echo "Waiting for container $container_name to exit..."
    docker wait $container_name
    echo "Container $container_name exited"
    docker remove $container_name
}

cleanup() {
    if [ $(is_primary_cloud) -eq 0 ]; then
        wait_for_exit primary_cloud
    fi
    wait_for_exit edge_server
    local clients_per_machine=$(jq -r '.clients_per_machine' config/params.json)
    for i in $(seq 0 $(($clients_per_machine - 1))); do
        wait_for_exit client_$i
    done

    echo "All containers exited"
}

main() {
    launch_nodes
    cleanup
}

main









