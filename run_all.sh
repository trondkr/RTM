# -----------------------------
#
# This script runs all of the light calculatons (close to) simultaneously. 
# The initialization between each new calculation is paused for 120 seconds 
# to prevent the python scripts from reading the same resources 
# at the same time.
#
# Trond Kristiansen, 22.02.2024
# -----------------------------

# Model: MPI-ESM1-2-LR
declare -a source_ids=("MPI-ESM1-2-LR")
declare -a member_ids=("r4i1p1f1" "r6i1p1f1" "r10i1p1f1" "r2i1p1f1" "r5i1p1f1")

for source_id in "${source_ids[@]}"; do
    for member_id in "${member_ids[@]}"; do
       
        echo "python CMIP6_light.py -m ""${source_id}"" -i ""${member_id}"" > "logs/${source_id}_${member_id}.txt""
        nohup python CMIP6_light.py -m ${source_id} -i ${member_id} > "logs/${source_id}_${member_id}.txt" &
        sleep 60
    done
done

# Model: UKESM1-0-LL
declare -a source_ids=("UKESM1-0-LL") 
declare -a member_ids=("r1i1p1f2" "r2i1p1f2" "r3i1p1f2" "r4i1p1f2" "r8i1p2f1") 
for source_id in "${source_ids[@]}"; do
    for member_id in "${member_ids[@]}"; do
       
        echo "python CMIP6_light.py -m ""${source_id}"" -i ""${member_id}"" > "logs/${source_id}_${member_id}.txt""
        nohup python CMIP6_light.py -m ${source_id} -i ${member_id} > "logs/${source_id}_${member_id}.txt" &
        sleep 60
    done
done

# Model: CanESM5
declare -a source_ids=("CanESM5") OK
declare -a member_ids=("r1i1p1f1" "r2i1p2f1" "r3i1p2f1" "r4i1p1f1" "r4i1p1f1" "r7i1p2f1" "r9i1p2f1" "r10i1p1f1" "r10i1p2f1" )

for source_id in "${source_ids[@]}"; do
    for member_id in "${member_ids[@]}"; do
       
        echo "python CMIP6_light.py -m ""${source_id}"" -i ""${member_id}"" > "logs/${source_id}_${member_id}.txt""
        nohup python CMIP6_light.py -m ${source_id} -i ${member_id} > "logs/${source_id}_${member_id}.txt" &
        sleep 60
    done
done

# Model: CMCC-ESM2
declare -a source_ids=("CMCC-ESM2") 
declare -a member_ids=("r1i1p1f1") 

for source_id in "${source_ids[@]}"; do
    for member_id in "${member_ids[@]}"; do
       
        echo "python CMIP6_light.py -m ""${source_id}"" -i ""${member_id}"" > "logs/${source_id}_${member_id}.txt""
        nohup python CMIP6_light.py -m ${source_id} -i ${member_id} > "logs/${source_id}_${member_id}.txt" &
        sleep 60
    done
done

# Model: MPI-ESM1-2-HR
declare -a source_ids=("MPI-ESM1-2-HR")
declare -a member_ids=("r1i1p1f1" "r2i1p1f1")

for source_id in "${source_ids[@]}"; do
    for member_id in "${member_ids[@]}"; do
       
        echo "python CMIP6_light.py -m ""${source_id}"" -i ""${member_id}"" > "logs/${source_id}_${member_id}.txt""
        nohup python CMIP6_light.py -m ${source_id} -i ${member_id} > "logs/${source_id}_${member_id}.txt" &
        sleep 60
    done
done
