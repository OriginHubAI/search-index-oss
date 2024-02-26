#!/bin/bash

# abort when a command returns non-zero value
set -e

cmd_list=()

# add float vector tests
metrics=("L2" "IP" "Cosine")
for metric in "${metrics[@]}"; do
    cmd_list+=( "./boost_ut_test_vector_index --metric ${metric} --use_default_params 1"
                "./boost_ut_test_vector_index --metric ${metric} --use_default_params 1 --filter_out_mod 3"
                )
done

# add binary vector tests
bin_metrics=("Hamming" "Jaccard")
for metric in "${bin_metrics[@]}"; do
    cmd_list+=( "./boost_ut_test_vector_index --metric ${metric} --data_dim 264 --num_data 4000 --index_types BinaryFLAT"
                "./boost_ut_test_vector_index --metric ${metric} --data_dim 264 --num_data 4000 --index_types BinaryIVF"
                "./boost_ut_test_vector_index --metric ${metric} --data_dim 264 --num_data 4000 --index_types BinaryHNSW"
                "./boost_ut_test_vector_index --metric ${metric} --data_dim 264 --num_data 4000 --index_types BinaryFLAT --filter_out_mod 5"
                "./boost_ut_test_vector_index --metric ${metric} --data_dim 264 --num_data 4000 --index_types BinaryIVF  --filter_out_mod 5"
                "./boost_ut_test_vector_index --metric ${metric} --data_dim 264 --num_data 4000 --index_types BinaryHNSW --filter_out_mod 5"
                )
done

for cmd in "${cmd_list[@]}"; do
    echo "## Executing $cmd"
    eval $cmd
    # clean up all the temp files
    rm -rf /tmp/test_vector_index_index_*
done
