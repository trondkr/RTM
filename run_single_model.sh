mkdir logs
python CMIP6_light.py -m MPI-ESM1-2-LR -i r4i1p1f1 --create_forcing > "logs/single_run_MPI-ESM1-2-LR_createforcing.txt"
python CMIP6_light.py -m MPI-ESM1-2-LR -i r4i1p1f1  > "logs/single_run_MPI-ESM1-2-LR_modelrun.txt"