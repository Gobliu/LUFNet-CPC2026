#!/bin/bash

file2='results/baseline_run'
cat $file2/log  > $file2/log_combine
./show_results.sh $file2/log_combine

python loss_weight.py load_file_tau0.1.dict baseline_run_ 0,1/8,0,1/4,0,1/2,0,1 "dpt=180000; ai tau =0.05; batch size 16; lr 1e-5; poly deg=4"  
