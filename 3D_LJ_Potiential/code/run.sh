#!/bin/bash

file2='results/traj_len08ws08tau0.05ngrid12api0lw8421ew1repw10poly4l_dpt180000'
cat $file2/log  > $file2/log_combine
./show_results.sh $file2/log_combine

python loss_weight.py load_file_tau0.1.dict traj8ws8tau0.05ngrid12w8421ew1repw10poly4l_ 0,1/8,0,1/4,0,1/2,0,1 "dpt=180000; ai tau =0.05; batch size 16; lr 1e-5; poly deg=4"  
 

