#!/bin/sh

host_name="gauss"

scp ${host_name}:./code/linear_memory/tests/data/*.pt ./tests/data/

echo "Fetched data from $host_name!"

# use chmod +x push_gauss.sh to make executable
