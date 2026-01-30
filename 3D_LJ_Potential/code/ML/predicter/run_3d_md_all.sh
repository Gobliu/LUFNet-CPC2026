#!/bin/bash

# compute rdf and then use to plot rdf curve
python rdf.py 64 0.85 0.9 3 1000 0.001 1000 20 None 180000 l > log/n64rho0.85T0.9t0.35_rdf_t100
# split 1000 samples into 5 groups, compute rdf for each group, and then use to plot mean/std across 5 groups
python rdf.py 64 0.85 0.9 3 200 0.001 1000 20 None 180000 l  > log/n64rho0.85T0.9t0.35s200_rdf_t100

# compute rdf and then use to plot rdf curve
python rdf.py 128 0.85 0.9 3 1000 0.001 1000 20 None 180000 l >  log/n128rho0.85T0.9t0.35_rdf_t100
# split 1000 samples into 5 groups, compute rdf for each group, and then use to plot mean/std across 5 groups
python rdf.py 128 0.85 0.9 3 200 0.001 1000 20 None 180000 l > log/n128rho0.85T0.9t0.35s200_rdf_t100

# compute rdf and then use to plot rdf curve
python rdf.py 256 0.85 0.9 3 1000 0.001 1000 20 None 180000 l >  log/256rho0.85T0.9t0.35_rdf_t100
# split 1000 samples into 5 groups, compute rdf for each group, and then use to plot mean/std across 5 groups
python rdf.py 256 0.85 0.9 3 200 0.001 1000 20 None 180000 l >  log/n256rho0.85T0.9t0.35s200_rdf_t100

# compute energy and then use to plot energy curve
python e_conserve.py 64 0.85 0.9 3 0.001 1000 20 None 1000 l > log/n64rho0.85T0.9t0.35_e_t100
# compute energy and then use to plot energy curve
python e_conserve.py 128 0.85 0.9 3 0.001 1000 20 None 1000 l > log/n128rho0.85T0.9t0.35_e_t100
# compute energy and then use to plot energy curve
python e_conserve.py 256 0.85 0.9 3 0.001 1000 20 None 1000 l > log/n256rho0.85T0.9t0.35_e_t100
