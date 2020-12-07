#!/bin/sh

host_name="gauss"

scp ${host_name}:./code/linear_memory/examples/gene_expression/learn_linear_ode_minibatch.pt ./examples/gene_expression/

echo "Fetched data from $host_name!"

# use chmod +x push_gauss.sh to make executable
