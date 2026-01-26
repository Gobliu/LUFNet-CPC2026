#!/bin/bash

python plot_rdf.py 64 0.85 0.9 3 0.05 20 065 180000 l
python plot_rdf.py 128 0.85 0.9 3 0.05 20 065 180000 l
python plot_rdf.py 256 0.85 0.9 3 0.05 20 065 180000 l
#
python save_rdf_all.py 64 0.85 3 20 065 180000 l 200
python save_rdf_all.py 128 0.85 3 20 065 180000 l 200
python save_rdf_all.py 256 0.85 3 20 065 180000 l 200

python save_e_mean_all.py 64 0.85 3 20 065 180000 l 200
python save_e_mean_all.py 128 0.85 3 20 065 180000 l 200
python save_e_mean_all.py 256 0.85 3 20 065 180000 l 200
#
python save_Cv_all.py 64 0.85 3 20 065 180000 l 200
python save_Cv_all.py 128 0.85 3 20 065 180000 l 200
python save_Cv_all.py 256 0.85 3 20 065 180000 l 200

mkdir ../../analysis/3d/LUF065
mv ../../analysis/3d/*txt ../../analysis/3d/LUF065/

python plot_rdf_all_npar_list.py 065 2.89 2.97 1.286 1.315
python plot_e_cv_all_npar_list.py 065 -0.005 0.14 -0.005 1.6
