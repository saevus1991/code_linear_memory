#!/bin/sh

host_name="gauss"

scp ./examples/gene_expression/data.npz ${host_name}:./code/linear_memory/examples/gene_expression/data.npz

echo "Fetched data from $host_name!"

# use chmod +x push_data.sh to make executable
