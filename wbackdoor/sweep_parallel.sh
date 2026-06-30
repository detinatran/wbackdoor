#!/usr/bin/env bash
# Chạy full sweep 5 theta x 4 rho = 20 job, chia đều lên N GPU chạy song song.
# Mỗi job = 1 lần train_backdoor.py với (theta,rho) riêng, ghi results JSON riêng.
# Dùng: bash sweep_parallel.sh
set -u

CODE_DIR=~/wbackdoor/code
OUT=$CODE_DIR/sweep_out_full_v2     # v2 = metric ASR calibrated mới (giữ kết quả cũ không bị đè)
CFG=configs/attack.yaml
EPOCHS=200
THETAS=(20 30 40 50 60)
RHOS=(0.01 0.02 0.05 0.1)
NGPU=2                         # chỉ dùng GPU 0,1 (round-robin tự gán CUDA_VISIBLE_DEVICES=0/1)

cd "$CODE_DIR" || exit 1
mkdir -p "$OUT"

# build danh sách job
JOBS=()
for rho in "${RHOS[@]}"; do
  for th in "${THETAS[@]}"; do
    JOBS+=("$th:$rho")
  done
done
echo "Tổng ${#JOBS[@]} job, chia lên $NGPU GPU, mỗi job $EPOCHS epoch."

# phát job theo round-robin lên các GPU; mỗi GPU chạy chuỗi job của nó nền
launch_gpu () {
  local gpu=$1; shift
  local mine=("$@")
  for spec in "${mine[@]}"; do
    th=${spec%%:*}; rho=${spec##*:}
    tag="th${th}_rho${rho}"
    echo "[gpu$gpu] start $tag"
    CUDA_VISIBLE_DEVICES=$gpu python3 -u - "$th" "$rho" "$tag" <<'PY' >> "$OUT/log_gpu${gpu}.log" 2>&1
import sys, json, yaml, copy, os
from train_backdoor import train
th, rho, tag = float(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
cfg = yaml.safe_load(open("configs/attack.yaml"))
cfg = copy.deepcopy(cfg)
cfg["theta_max_deg"] = th; cfg["rho"] = rho; cfg["epochs"] = 200
print(f"=== TRAIN {tag} theta={th} rho={rho} ===", flush=True)
_, res = train(cfg)
res["_theta_max_deg"] = th; res["_rho"] = rho
json.dump(res, open(os.path.join(os.path.expanduser("~"),
          "wbackdoor/code/sweep_out_full_v2", f"res_{tag}.json"), "w"), indent=2)
print(f"=== DONE {tag} ===", flush=True)
PY
  done
  echo "[gpu$gpu] ALL DONE"
}

# round-robin phân job
declare -a BUCK
for i in "${!JOBS[@]}"; do
  g=$(( i % NGPU ))
  BUCK[$g]="${BUCK[$g]:-} ${JOBS[$i]}"
done

for g in $(seq 0 $((NGPU-1))); do
  read -ra arr <<< "${BUCK[$g]:-}"
  [ ${#arr[@]} -eq 0 ] && continue
  launch_gpu "$g" "${arr[@]}" &
done
wait
echo "==== TẤT CẢ JOB XONG. Kết quả res_*.json trong $OUT ===="
