#!/usr/bin/env bash

set -euo pipefail

BIN=${BIN:-./evenodd_parix}
OUT_CSV=${OUT_CSV:-results_parix_alloc_comparison.csv}

# 测试参数
REPEATS=${REPEATS:-20000}
PACKETSIZE=${PACKETSIZE:-1024}
KS=(${KS:-5 7 11 13})
UPDATES=(${UPDATES:-1024 2048 4096 8192 16384 32768})

# 四种方案：顺序地址分配、优化地址分配、PARIX、PARIX+优化地址分配
declare -A MODES
MODES[seq]="-a sequential -X off"
MODES[opt]="-a auto -X off"
MODES[parix]="-a sequential -X parix"
MODES[parix_opt]="-a auto -X parix+alloc"

# 允许自定义 PARIX 目录根路径
PARIX_ROOT=${PARIX_ROOT:-./parix_runs}
mkdir -p "$PARIX_ROOT"

# 可选：磁盘列表（如果不提供，将走回退路径/合成 I/O）
DISKS=(${DISKS:-})

echo "检查可执行文件..."
if [[ ! -x "$BIN" ]]; then
  echo "可执行文件 $BIN 不存在或不可执行。请先编译："
  echo "  gcc -O3 -Wall -Wextra -std=c11 -D_GNU_SOURCE -march=native -o evenodd_parix \\
      evenodd_parix.c alloc_strategy.c parix_module.c -luring -lpthread -lm"
  echo "或运行: ./compile.sh"
  exit 1
fi

# 如果提供了 DISKS，则生成 safe_disks.txt
if [[ ${#DISKS[@]} -gt 0 ]]; then
  echo "生成 safe_disks.txt ..."
  : > safe_disks.txt
  FOUND=0
  for d in "${DISKS[@]}"; do
    if [[ -b "$d" ]]; then
      echo "$d" >> safe_disks.txt
      ((FOUND++)) || true
    fi
  done
  echo "写入了 $FOUND 个存在的块设备到 safe_disks.txt"
fi

# 输出表头
echo "mode,k,update_bytes,alloc,packetsize_bytes,repeats,elapsed_s,IOPS,throughput_MBps,success" > "$OUT_CSV"

now_ns() { date +%s%N; }

run_case() {
  local mode_key="$1"; shift
  local k="$1"; shift
  local u="$1"; shift
  local args="${MODES[$mode_key]}"

  # 为 PARIX 运行指定独立目录，避免相互污染
  local parix_dir="$PARIX_ROOT/${mode_key}_k${k}_u${u}_$(date +%s)"
  mkdir -p "$parix_dir"

  # 如果模式包含 PARIX，附加 -L 目录
  if [[ "$args" == *"parix"* ]]; then
    args+=" -L $parix_dir"
  fi

  # 提取 alloc 策略文字
  local alloc="sequential"
  if [[ "$args" == *"-a auto"* ]]; then alloc="auto"; fi

  local success=1
  local output=""
  if ! output=$($BIN -k "$k" -m 2 -w "$k" -p "$PACKETSIZE" -u "$u" -n "$REPEATS" $args 2>/dev/null); then
    success=0
  fi

  local line
  line=$(printf "%s\n" "$output" | grep '^\[RESULT\] ' || true)
  local elapsed_s="0" IOPS="0" THROUGHPUT="0"
  if [[ -n "$line" ]]; then
    elapsed_s=$(awk '{for(i=1;i<=NF;i++) if($i ~ /^elapsed=/){gsub(/elapsed=|s,?/,"",$i); print $i}}' <<<"$line")
    IOPS=$(awk '{for(i=1;i<=NF;i++) if($i ~ /^IOPS=/){gsub(/IOPS=|,?/,"",$i); print $i}}' <<<"$line")
    THROUGHPUT=$(awk '{for(i=1;i<=NF;i++) if($i ~ /^throughput=/){gsub(/throughput=|MB\/s,?/,"",$i); print $i}}' <<<"$line")
  else
    success=0
  fi

  echo "$mode_key,$k,$u,$alloc,$PACKETSIZE,$REPEATS,$elapsed_s,$IOPS,$THROUGHPUT,$success" >> "$OUT_CSV"
}

echo "开始测试... 输出: $OUT_CSV"
for k in "${KS[@]}"; do
  # EVENODD 要求 w>=k-1，这里取 w=k
  for u in "${UPDATES[@]}"; do
    run_case seq "$k" "$u"
    run_case opt "$k" "$u"
    run_case parix "$k" "$u"
    run_case parix_opt "$k" "$u"
  done
done

echo "测试完成。结果已写入 $OUT_CSV"


