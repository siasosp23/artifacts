# invoking Sia with p=-0.5, lambda_n=1.1, interval=60
export OUT_DIR="/tmp/sia-saturn"
mkdir -p ${OUT_DIR}
python multi_simulator.py ../workloads_v2/saturn/ --policy=sia --policy_p_val=-0.5 --mip_lambda_a=0.01 --mip_lambda_n=1.1 --interval=60 --project_throughput --share_max_replicas --output=${OUT_DIR}

# invoke Gavel with interval=360
export OUT_DIR="/tmp/gavel-saturn"
mkdir -p ${OUT_DIR}
python multi_simulator.py ../workloads_v2/saturn/ --policy=gavel --interval=360 --output=${OUT_DIR}
