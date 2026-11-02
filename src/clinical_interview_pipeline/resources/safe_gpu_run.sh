#!/usr/bin/env bash
# Run a GPU command with resource checks, power limits, a watchdog, and logs.

set -Eeuo pipefail

# Usage:
#   CUDA_VISIBLE_DEVICES="0,1,2" \
#   POWER_LIMIT_WATTS=250 \
#   MAX_GPU_TEMP=72 \
#   COOLDOWN_TEMP=60 \
#   MAX_CPU_TEMP=110 \
#   CPU_COOLDOWN_TEMP=100 \
#   START_MAX_CPU_BUSY_PERCENT=70 \
#   MIN_AVAILABLE_MEM_GIB=16 \
#   MAX_SWAP_USED_PERCENT=80 \
#   WATCH_DISK_PATH=/private/clinical-workspace \
#   MIN_FREE_DISK_GIB=20 \
#   CPU_THREADS_PER_JOB=4 \
#   RUN_TAG=my_run \
#   RUN_BASE_DIR=/private/clinical-workspace/runs/safety \
#   bash safe_gpu_run.sh -- <command>
#
POWER_LIMIT_WATTS="${POWER_LIMIT_WATTS:-250}"
MAX_GPU_TEMP="${MAX_GPU_TEMP:-72}"
COOLDOWN_TEMP="${COOLDOWN_TEMP:-60}"

MAX_CPU_TEMP="${MAX_CPU_TEMP:-110}"
CPU_COOLDOWN_TEMP="${CPU_COOLDOWN_TEMP:-100}"
REQUIRE_CPU_TEMP_SENSOR="${REQUIRE_CPU_TEMP_SENSOR:-0}"

START_MAX_CPU_BUSY_PERCENT="${START_MAX_CPU_BUSY_PERCENT:-70}"
START_MAX_LOAD_PER_CPU="${START_MAX_LOAD_PER_CPU:-0.80}"
STARTUP_WAIT_TIMEOUT="${STARTUP_WAIT_TIMEOUT:-1800}"
STARTUP_POLL_INTERVAL="${STARTUP_POLL_INTERVAL:-15}"

MIN_AVAILABLE_MEM_GIB="${MIN_AVAILABLE_MEM_GIB:-16}"
MAX_SWAP_USED_PERCENT="${MAX_SWAP_USED_PERCENT:-80}"

WATCH_DISK_PATH="${WATCH_DISK_PATH:-}"
MIN_FREE_DISK_GIB="${MIN_FREE_DISK_GIB:-20}"

# Runtime CPU busy enforcement is off by default. Set a positive value to
# enable it.
MAX_CPU_BUSY_PERCENT="${MAX_CPU_BUSY_PERCENT:-0}"

SAFETY_BREACH_SAMPLES="${SAFETY_BREACH_SAMPLES:-3}"
WATCHDOG_INTERVAL="${WATCHDOG_INTERVAL:-5}"

GPU_LOG_INTERVAL="${GPU_LOG_INTERVAL:-20}"
SYS_LOG_INTERVAL="${SYS_LOG_INTERVAL:-20}"

CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-4}"
FFMPEG_THREADS="${FFMPEG_THREADS:-4}"
NICE_LEVEL="${NICE_LEVEL:-5}"
IONICE_CLASS="${IONICE_CLASS:-2}"
IONICE_LEVEL="${IONICE_LEVEL:-5}"

RUN_TAG="${RUN_TAG:-safe_gpu_run}"
RUN_BASE_DIR="${RUN_BASE_DIR:-}"
REQUIRE_POWER_CAP="${REQUIRE_POWER_CAP:-1}"
RESTORE_POWER_LIMITS="${RESTORE_POWER_LIMITS:-0}"

if [[ "${1:-}" != "--" ]]; then
    echo "ERROR: Usage: bash $0 -- <command>"
    exit 2
fi
shift

if [[ "$#" -eq 0 ]]; then
    echo "ERROR: No command provided."
    exit 2
fi

if [[ -z "$RUN_BASE_DIR" || -z "$WATCH_DISK_PATH" ]]; then
    echo "ERROR: Set RUN_BASE_DIR and WATCH_DISK_PATH to private workspace paths." >&2
    exit 2
fi
if [[ ! "$RUN_TAG" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo "ERROR: RUN_TAG may contain only letters, digits, dots, underscores, and hyphens." >&2
    exit 2
fi

ts() { date +"%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(ts)] $*"; }

RUN_DIR="$RUN_BASE_DIR/${RUN_TAG}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"

CMD_LOG="$RUN_DIR/command.log"
GPU_LOG="$RUN_DIR/gpu.log"
SYS_LOG="$RUN_DIR/system.log"
ENV_LOG="$RUN_DIR/env.log"
STATUS_LOG="$RUN_DIR/status.log"
FINAL_LOG="$RUN_DIR/final_snapshot.log"
SAFETY_STOP_FILE="$RUN_DIR/SAFETY_STOP"

COMMAND_PID=""
GPU_LOG_PID=""
SYS_LOG_PID=""
WATCHDOG_PID=""

declare -A ORIGINAL_POWER_LIMITS=()

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
else
    mapfile -t GPU_IDS < <(
        nvidia-smi --query-gpu=index --format=csv,noheader,nounits
    )
fi

# Limit common CPU thread pools inherited by Python and child processes.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$CPU_THREADS_PER_JOB}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$CPU_THREADS_PER_JOB}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$CPU_THREADS_PER_JOB}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-$CPU_THREADS_PER_JOB}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-$CPU_THREADS_PER_JOB}"
export BLIS_NUM_THREADS="${BLIS_NUM_THREADS:-$CPU_THREADS_PER_JOB}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export FFMPEG_THREADS

validate_numeric_settings() {
    local integer_vars=(
        POWER_LIMIT_WATTS MAX_GPU_TEMP COOLDOWN_TEMP
        MAX_CPU_TEMP CPU_COOLDOWN_TEMP
        START_MAX_CPU_BUSY_PERCENT STARTUP_WAIT_TIMEOUT
        STARTUP_POLL_INTERVAL MIN_AVAILABLE_MEM_GIB
        MAX_SWAP_USED_PERCENT MIN_FREE_DISK_GIB
        MAX_CPU_BUSY_PERCENT SAFETY_BREACH_SAMPLES
        WATCHDOG_INTERVAL GPU_LOG_INTERVAL SYS_LOG_INTERVAL
        CPU_THREADS_PER_JOB FFMPEG_THREADS NICE_LEVEL
        IONICE_CLASS IONICE_LEVEL
    )
    local name value

    for name in "${integer_vars[@]}"; do
        value="${!name}"
        if [[ ! "$value" =~ ^[0-9]+$ ]]; then
            log "ERROR: $name must be a non-negative integer; got '$value'."
            exit 2
        fi
    done

    if [[ ! "$START_MAX_LOAD_PER_CPU" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        log "ERROR: START_MAX_LOAD_PER_CPU must be numeric."
        exit 2
    fi

    if (( COOLDOWN_TEMP > MAX_GPU_TEMP )); then
        log "ERROR: COOLDOWN_TEMP must not exceed MAX_GPU_TEMP."
        exit 2
    fi

    if (( CPU_COOLDOWN_TEMP > MAX_CPU_TEMP )); then
        log "ERROR: CPU_COOLDOWN_TEMP must not exceed MAX_CPU_TEMP."
        exit 2
    fi

    if (( MAX_SWAP_USED_PERCENT > 100 )); then
        log "ERROR: MAX_SWAP_USED_PERCENT must be between 0 and 100."
        exit 2
    fi
}

validate_gpu_ids() {
    local gpu
    if [[ "${#GPU_IDS[@]}" -eq 0 ]]; then
        log "ERROR: No GPUs selected."
        exit 5
    fi

    for gpu in "${GPU_IDS[@]}"; do
        gpu="${gpu//[[:space:]]/}"
        if [[ ! "$gpu" =~ ^[0-9]+$ ]]; then
            log "ERROR: Invalid physical GPU index: '$gpu'"
            exit 5
        fi
        if ! nvidia-smi -i "$gpu" --query-gpu=index \
            --format=csv,noheader,nounits >/dev/null 2>&1; then
            log "ERROR: Physical GPU index $gpu does not exist or is unavailable."
            exit 5
        fi
    done
}

pin_cuda_devices_by_uuid() {
    # CUDA ordinals can differ from nvidia-smi indices. Pass UUIDs to the
    # command so the GPUs we monitor are the GPUs the command actually uses.
    local gpu uuid
    local -a gpu_uuids=()

    for gpu in "${GPU_IDS[@]}"; do
        gpu="${gpu//[[:space:]]/}"
        uuid="$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader)"
        if [[ ! "$uuid" =~ ^GPU-[0-9A-Fa-f-]+$ ]]; then
            log "ERROR: Cannot resolve physical GPU $gpu to a UUID."
            exit 5
        fi
        gpu_uuids+=("$uuid")
    done

    local IFS=,
    CUDA_VISIBLE_DEVICES="${gpu_uuids[*]}"
    export CUDA_VISIBLE_DEVICES
    log "Pinned physical GPU indices ${GPU_IDS[*]} by UUID for the command."
}

gpu_query() {
    local query="$1"
    local gpu
    for gpu in "${GPU_IDS[@]}"; do
        gpu="${gpu//[[:space:]]/}"
        nvidia-smi -i "$gpu" \
            --query-gpu="$query" \
            --format=csv,noheader,nounits
    done
}

get_hottest_gpu_temp() {
    gpu_query "temperature.gpu" |
        awk '
            BEGIN { max = -1 }
            NF && $1 ~ /^[0-9]+$/ {
                if ($1 > max) max = $1
            }
            END { print max }
        '
}

get_hottest_cpu_temp() {
    # Prefer kernel CPU package temperature when available.
    # This avoids parsing lm-sensors threshold fields such as "crit = +100.0°C".
    local z type temp_raw value
    local max="-1"

    for z in /sys/class/thermal/thermal_zone*; do
        [[ -d "$z" ]] || continue
        type="$(cat "$z/type" 2>/dev/null || true)"
        temp_raw="$(cat "$z/temp" 2>/dev/null || true)"

        if [[ "$type" == "x86_pkg_temp" && "$temp_raw" =~ ^[0-9]+$ ]]; then
            value="$(awk -v x="$temp_raw" 'BEGIN { printf "%.0f", x / 1000 }')"
            if [[ "$value" =~ ^[0-9]+$ ]] && (( value > max )); then
                max="$value"
            fi
        fi
    done

    if [[ "$max" -ge 0 ]]; then
        echo "$max"
        return
    fi

    # Fallback to hwmon coretemp/k10temp CPU package/core sensors only.
    # Ignore motherboard chips such as nct6798 AUXTIN/SYSTIN and non-CPU devices.
    local f hwmon_dir chip label
    for f in /sys/class/hwmon/hwmon*/temp*_input; do
        [[ -f "$f" ]] || continue

        hwmon_dir="$(dirname "$f")"
        chip="$(cat "$hwmon_dir/name" 2>/dev/null || true)"
        label="$(cat "${f%_input}_label" 2>/dev/null || true)"
        temp_raw="$(cat "$f" 2>/dev/null || true)"

        case "$chip" in
            coretemp|k10temp)
                ;;
            *)
                continue
                ;;
        esac

        if [[ "$label" =~ ^(Package id|Core|Tctl|Tdie) ]] && [[ "$temp_raw" =~ ^[0-9]+$ ]]; then
            value="$(awk -v x="$temp_raw" 'BEGIN { printf "%.0f", x / 1000 }')"
            if [[ "$value" =~ ^[0-9]+$ ]] && (( value > max )); then
                max="$value"
            fi
        fi
    done

    if [[ "$max" -ge 0 ]]; then
        echo "$max"
        return
    fi

    # Last-resort lm-sensors parser:
    # extract only the first +XX°C value from CPU-relevant lines, not high/crit fields.
    if command -v sensors >/dev/null 2>&1; then
        sensors 2>/dev/null |
            awk '
                /Package id [0-9]+:|Physical id [0-9]+:|Core [0-9]+:|Tctl:|Tdie:|CPU Temp:/ {
                    if (match($0, /\+[0-9]+([.][0-9]+)?°?C/)) {
                        value = substr($0, RSTART, RLENGTH)
                        gsub(/[^0-9.]/, "", value)
                        if (value != "" && value + 0 > max) {
                            max = value + 0
                        }
                    }
                }
                END {
                    if (max == "") print -1
                    else printf "%.0f\n", max
                }
            '
        return
    fi

    echo -1
}

cpu_counters() {
    awk '
        /^cpu / {
            idle = $5 + $6
            total = 0
            for (i = 2; i <= 9; i++) total += $i
            print total, idle
            exit
        }
    ' /proc/stat
}

get_cpu_busy_percent() {
    local total1 idle1 total2 idle2 delta_total delta_idle
    read -r total1 idle1 < <(cpu_counters)
    sleep 1
    read -r total2 idle2 < <(cpu_counters)

    delta_total=$((total2 - total1))
    delta_idle=$((idle2 - idle1))

    if (( delta_total <= 0 )); then
        echo -1
        return
    fi

    awk -v total="$delta_total" -v idle="$delta_idle" \
        'BEGIN { printf "%.0f\n", 100 * (total - idle) / total }'
}

get_load_per_cpu() {
    local load1 cpus
    load1="$(awk '{print $1}' /proc/loadavg)"
    cpus="$(nproc)"
    awk -v load_value="$load1" -v cpus="$cpus" \
        'BEGIN { if (cpus < 1) cpus = 1; printf "%.2f\n", load_value / cpus }'
}

get_available_mem_gib() {
    awk '
        /MemAvailable:/ {
            printf "%.0f\n", $2 / 1024 / 1024
            exit
        }
    ' /proc/meminfo
}

get_swap_used_percent() {
    awk '
        /SwapTotal:/ { total = $2 }
        /SwapFree:/  { free = $2 }
        END {
            if (total <= 0) print 0
            else printf "%.0f\n", 100 * (total - free) / total
        }
    ' /proc/meminfo
}

get_free_disk_gib() {
    df -Pk "$WATCH_DISK_PATH" 2>/dev/null |
        awk 'NR == 2 { printf "%.0f\n", $4 / 1024 / 1024 }'
}

stop_pid() {
    local pid="${1:-}"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    fi
}

terminate_command() {
    if [[ -z "${COMMAND_PID:-}" ]] || ! kill -0 "$COMMAND_PID" 2>/dev/null; then
        return
    fi

    log "Stopping command process group..."

    kill -TERM -- "-$COMMAND_PID" 2>/dev/null || \
        kill -TERM "$COMMAND_PID" 2>/dev/null || true

    for _ in {1..15}; do
        if ! kill -0 "$COMMAND_PID" 2>/dev/null; then
            return
        fi
        sleep 1
    done

    log "Command did not stop after SIGTERM; sending SIGKILL."
    kill -KILL -- "-$COMMAND_PID" 2>/dev/null || \
        kill -KILL "$COMMAND_PID" 2>/dev/null || true
}

restore_power_limits() {
    [[ "$RESTORE_POWER_LIMITS" == "1" ]] || return 0
    [[ "${#ORIGINAL_POWER_LIMITS[@]}" -gt 0 ]] || return 0

    local gpu original
    log "Restoring original GPU power limits where possible..."
    for gpu in "${!ORIGINAL_POWER_LIMITS[@]}"; do
        original="${ORIGINAL_POWER_LIMITS[$gpu]}"
        sudo -n nvidia-smi -i "$gpu" -pl "$original" \
            >> "$ENV_LOG" 2>&1 || true
    done
}

cleanup() {
    local original_rc=$?

    stop_pid "${WATCHDOG_PID:-}"
    stop_pid "${GPU_LOG_PID:-}"
    stop_pid "${SYS_LOG_PID:-}"

    if [[ "$original_rc" -ne 0 ]]; then
        terminate_command
    fi

    restore_power_limits
    return "$original_rc"
}

trap cleanup EXIT
trap 'log "Interrupted."; exit 130' INT
trap 'log "Termination requested."; exit 143' TERM

write_env_snapshot() {
    {
        echo "===== $(ts) ====="
        echo "RUN_DIR=$RUN_DIR"
        echo "RUN_TAG=$RUN_TAG"
        echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<not set>}"
        echo "SELECTED_GPU_IDS=${GPU_IDS[*]}"
        echo "POWER_LIMIT_WATTS=$POWER_LIMIT_WATTS"
        echo "MAX_GPU_TEMP=$MAX_GPU_TEMP"
        echo "COOLDOWN_TEMP=$COOLDOWN_TEMP"
        echo "MAX_CPU_TEMP=$MAX_CPU_TEMP"
        echo "CPU_COOLDOWN_TEMP=$CPU_COOLDOWN_TEMP"
        echo "REQUIRE_CPU_TEMP_SENSOR=$REQUIRE_CPU_TEMP_SENSOR"
        echo "START_MAX_CPU_BUSY_PERCENT=$START_MAX_CPU_BUSY_PERCENT"
        echo "START_MAX_LOAD_PER_CPU=$START_MAX_LOAD_PER_CPU"
        echo "STARTUP_WAIT_TIMEOUT=$STARTUP_WAIT_TIMEOUT"
        echo "MIN_AVAILABLE_MEM_GIB=$MIN_AVAILABLE_MEM_GIB"
        echo "MAX_SWAP_USED_PERCENT=$MAX_SWAP_USED_PERCENT"
        echo "WATCH_DISK_PATH=$WATCH_DISK_PATH"
        echo "MIN_FREE_DISK_GIB=$MIN_FREE_DISK_GIB"
        echo "MAX_CPU_BUSY_PERCENT=$MAX_CPU_BUSY_PERCENT"
        echo "SAFETY_BREACH_SAMPLES=$SAFETY_BREACH_SAMPLES"
        echo "CPU_THREADS_PER_JOB=$CPU_THREADS_PER_JOB"
        echo "FFMPEG_THREADS=$FFMPEG_THREADS"
        echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
        echo "MKL_NUM_THREADS=$MKL_NUM_THREADS"
        echo "OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS"
        echo "NUMEXPR_NUM_THREADS=$NUMEXPR_NUM_THREADS"
        echo "NICE_LEVEL=$NICE_LEVEL"
        echo "IONICE_CLASS=$IONICE_CLASS"
        echo "IONICE_LEVEL=$IONICE_LEVEL"
        echo
        echo "---- command ----"
        printf '%q ' "$@"
        echo
        echo
        echo "---- user / host / cwd ----"
        whoami || true
        hostname || true
        pwd || true
        echo
        echo "---- kernel ----"
        uname -a || true
        echo
        echo "---- CPU ----"
        lscpu || true
        echo
        echo "---- memory ----"
        free -h || true
        echo
        echo "---- sensors ----"
        sensors || true
        echo
        echo "---- disks ----"
        df -hT "$WATCH_DISK_PATH" || true
        echo
        echo "---- nvidia-smi -L ----"
        nvidia-smi -L || true
        echo
        echo "---- selected GPU status ----"
        local gpu
        for gpu in "${GPU_IDS[@]}"; do
            gpu="${gpu//[[:space:]]/}"
            nvidia-smi -i "$gpu" \
                --query-gpu=index,name,temperature.gpu,power.draw,power.limit,memory.used,memory.total \
                --format=csv || true
        done
    } > "$ENV_LOG" 2>&1
}

apply_power_cap() {
    local gpu original failed=0
    local -a uncapped_gpus=()

    for gpu in "${GPU_IDS[@]}"; do
        gpu="${gpu//[[:space:]]/}"

        original="$(
            nvidia-smi -i "$gpu" \
                --query-gpu=power.limit \
                --format=csv,noheader,nounits |
            awk 'NR == 1 { printf "%d", $1 }'
        )"
        if [[ ! "$original" =~ ^[0-9]+$ ]]; then
            log "ERROR: Cannot read the power limit for physical GPU $gpu."
            exit 3
        fi
        ORIGINAL_POWER_LIMITS["$gpu"]="$original"

        if (( original <= POWER_LIMIT_WATTS )); then
            log "Physical GPU $gpu is already capped at ${original} W."
        else
            uncapped_gpus+=("$gpu")
        fi
    done

    if [[ "${#uncapped_gpus[@]}" -eq 0 ]]; then
        return
    fi

    log "Requesting sudo access to cap ${#uncapped_gpus[@]} GPU(s)..."
    if ! sudo -v; then
        if [[ "$REQUIRE_POWER_CAP" == "1" ]]; then
            log "ERROR: sudo authentication failed; power cap cannot be guaranteed."
            exit 3
        fi
        log "WARNING: Continuing without a guaranteed power cap."
        return
    fi

    sudo nvidia-smi -pm 1 >> "$ENV_LOG" 2>&1 || true

    for gpu in "${uncapped_gpus[@]}"; do
        log "Setting physical GPU $gpu to ${POWER_LIMIT_WATTS} W."
        if ! sudo nvidia-smi -i "$gpu" -pl "$POWER_LIMIT_WATTS" \
            >> "$ENV_LOG" 2>&1; then
            failed=1
            log "WARNING: Could not set power limit on GPU $gpu."
        fi
    done

    if [[ "$failed" -eq 1 && "$REQUIRE_POWER_CAP" == "1" ]]; then
        log "ERROR: At least one requested GPU power cap failed."
        exit 4
    fi
}

wait_for_safe_start() {
    local start_epoch now elapsed
    local gpu_temp cpu_temp cpu_busy load_per_cpu
    local available_mem swap_used free_disk
    local safe

    start_epoch="$(date +%s)"
    log "Waiting for safe shared-server start conditions..."

    while true; do
        safe=1

        gpu_temp="$(get_hottest_gpu_temp || echo -1)"
        cpu_temp="$(get_hottest_cpu_temp || echo -1)"
        cpu_busy="$(get_cpu_busy_percent || echo -1)"
        load_per_cpu="$(get_load_per_cpu || echo -1)"
        available_mem="$(get_available_mem_gib || echo -1)"
        swap_used="$(get_swap_used_percent || echo -1)"
        free_disk="$(get_free_disk_gib || echo -1)"

        log "Preflight: GPU=${gpu_temp}°C CPU=${cpu_temp}°C CPU-busy=${cpu_busy}% load/core=${load_per_cpu} available-RAM=${available_mem}GiB swap-used=${swap_used}% disk-free=${free_disk}GiB"

        if [[ "$gpu_temp" =~ ^[0-9]+$ ]] && (( gpu_temp > COOLDOWN_TEMP )); then
            log "WAIT: selected GPU temperature exceeds ${COOLDOWN_TEMP}°C."
            safe=0
        fi

        if [[ "$cpu_temp" =~ ^[0-9]+$ ]] && (( cpu_temp >= 0 )); then
            if (( cpu_temp > CPU_COOLDOWN_TEMP )); then
                log "WAIT: CPU temperature exceeds ${CPU_COOLDOWN_TEMP}°C."
                safe=0
            fi
        elif [[ "$REQUIRE_CPU_TEMP_SENSOR" == "1" ]]; then
            log "WAIT: CPU temperature sensor unavailable and required."
            safe=0
        else
            log "NOTICE: CPU temperature sensor unavailable; continuing with other checks."
        fi

        if [[ "$cpu_busy" =~ ^[0-9]+$ ]] &&
           (( START_MAX_CPU_BUSY_PERCENT > 0 )) &&
           (( cpu_busy > START_MAX_CPU_BUSY_PERCENT )); then
            log "WAIT: CPU busy percentage exceeds ${START_MAX_CPU_BUSY_PERCENT}%."
            safe=0
        fi

        if awk -v current="$load_per_cpu" -v maximum="$START_MAX_LOAD_PER_CPU" \
            'BEGIN { exit !(maximum > 0 && current > maximum) }'; then
            log "WAIT: one-minute load per CPU exceeds ${START_MAX_LOAD_PER_CPU}."
            safe=0
        fi

        if [[ "$available_mem" =~ ^[0-9]+$ ]] &&
           (( available_mem < MIN_AVAILABLE_MEM_GIB )); then
            log "WAIT: available RAM is below ${MIN_AVAILABLE_MEM_GIB} GiB."
            safe=0
        fi

        if [[ "$swap_used" =~ ^[0-9]+$ ]] &&
           (( swap_used > MAX_SWAP_USED_PERCENT )); then
            log "WAIT: swap usage exceeds ${MAX_SWAP_USED_PERCENT}%."
            safe=0
        fi

        if [[ "$free_disk" =~ ^[0-9]+$ ]] &&
           (( free_disk < MIN_FREE_DISK_GIB )); then
            log "WAIT: free disk space is below ${MIN_FREE_DISK_GIB} GiB."
            safe=0
        fi

        if (( safe == 1 )); then
            log "Safe start conditions satisfied."
            break
        fi

        now="$(date +%s)"
        elapsed=$((now - start_epoch))
        if (( STARTUP_WAIT_TIMEOUT > 0 && elapsed >= STARTUP_WAIT_TIMEOUT )); then
            log "ERROR: Safe start conditions were not reached within ${STARTUP_WAIT_TIMEOUT}s."
            exit 74
        fi

        sleep "$STARTUP_POLL_INTERVAL"
    done
}

log_user_aggregates() {
    echo "---- aggregate resource use by user ----"
    ps -eo user=,%cpu=,rss= |
        awk '
            {
                cpu[$1] += $2
                rss[$1] += $3
                count[$1] += 1
            }
            END {
                for (user in cpu) {
                    printf "%-20s processes=%5d cpu=%8.1f%% rss=%10.1f MiB\n",
                           user, count[user], cpu[user], rss[user] / 1024
                }
            }
        ' |
        sort -k4 -nr || true
}

start_loggers() {
    log "Starting GPU logger: $GPU_LOG"
    (
        while true; do
            echo "===== $(ts) ====="
            for gpu in "${GPU_IDS[@]}"; do
                gpu="${gpu//[[:space:]]/}"
                nvidia-smi -i "$gpu" \
                    --query-gpu=index,name,temperature.gpu,pstate,utilization.gpu,memory.used,memory.total,power.draw,power.limit \
                    --format=csv,noheader,nounits || true
            done
            echo
            nvidia-smi pmon -c 1 2>/dev/null || true
            echo
            sleep "$GPU_LOG_INTERVAL"
        done
    ) >> "$GPU_LOG" 2>&1 &
    GPU_LOG_PID=$!

    log "Starting system logger: $SYS_LOG"
    (
        while true; do
            echo "===== $(ts) ====="

            echo "---- summary ----"
            echo "cpu_busy_percent=$(get_cpu_busy_percent || echo -1)"
            echo "load_per_cpu=$(get_load_per_cpu || echo -1)"
            echo "hottest_cpu_temp_c=$(get_hottest_cpu_temp || echo -1)"
            echo "available_mem_gib=$(get_available_mem_gib || echo -1)"
            echo "swap_used_percent=$(get_swap_used_percent || echo -1)"
            echo "free_disk_gib=$(get_free_disk_gib || echo -1)"
            echo

            echo "---- uptime / load ----"
            uptime || true
            cat /proc/loadavg || true
            echo "logical_cpus=$(nproc)"
            echo

            echo "---- temperatures ----"
            sensors || true
            echo

            echo "---- memory / swap ----"
            free -h || true
            echo

            echo "---- vmstat ----"
            vmstat 1 2 | tail -n 1 || true
            echo

            echo "---- disk ----"
            df -hT "$WATCH_DISK_PATH" || true
            echo

            if command -v iostat >/dev/null 2>&1; then
                echo "---- iostat ----"
                iostat -xz 1 1 || true
                echo
            fi

            log_user_aggregates
            echo

            echo "---- top CPU processes ----"
            ps -eo user:18,pid,ppid,%cpu,%mem,rss,etime,stat,cmd \
                --sort=-%cpu | head -n 35 || true
            echo

            echo "---- top RAM processes ----"
            ps -eo user:18,pid,ppid,%cpu,%mem,rss,etime,stat,cmd \
                --sort=-rss | head -n 35 || true
            echo

            sleep "$SYS_LOG_INTERVAL"
        done
    ) >> "$SYS_LOG" 2>&1 &
    SYS_LOG_PID=$!
}

start_safety_watchdog() {
    log "Starting safety watchdog."
    (
        local breach_count=0
        local reason=""
        local gpu_temp cpu_temp cpu_busy
        local available_mem swap_used free_disk

        while true; do
            reason=""

            gpu_temp="$(get_hottest_gpu_temp || echo -1)"
            cpu_temp="$(get_hottest_cpu_temp || echo -1)"
            available_mem="$(get_available_mem_gib || echo -1)"
            swap_used="$(get_swap_used_percent || echo -1)"
            free_disk="$(get_free_disk_gib || echo -1)"

            if [[ "$gpu_temp" =~ ^[0-9]+$ ]] && (( gpu_temp > MAX_GPU_TEMP )); then
                reason+="GPU temperature ${gpu_temp}°C > ${MAX_GPU_TEMP}°C; "
            fi

            if [[ "$cpu_temp" =~ ^[0-9]+$ ]] && (( cpu_temp >= 0 )); then
                if (( cpu_temp > MAX_CPU_TEMP )); then
                    reason+="CPU temperature ${cpu_temp}°C > ${MAX_CPU_TEMP}°C; "
                fi
            elif [[ "$REQUIRE_CPU_TEMP_SENSOR" == "1" ]]; then
                reason+="CPU temperature sensor unavailable; "
            fi

            if [[ "$available_mem" =~ ^[0-9]+$ ]] &&
               (( available_mem < MIN_AVAILABLE_MEM_GIB )); then
                reason+="available RAM ${available_mem}GiB < ${MIN_AVAILABLE_MEM_GIB}GiB; "
            fi

            if [[ "$swap_used" =~ ^[0-9]+$ ]] &&
               (( swap_used > MAX_SWAP_USED_PERCENT )); then
                reason+="swap used ${swap_used}% > ${MAX_SWAP_USED_PERCENT}%; "
            fi

            if [[ "$free_disk" =~ ^[0-9]+$ ]] &&
               (( free_disk < MIN_FREE_DISK_GIB )); then
                reason+="free disk ${free_disk}GiB < ${MIN_FREE_DISK_GIB}GiB; "
            fi

            if (( MAX_CPU_BUSY_PERCENT > 0 )); then
                cpu_busy="$(get_cpu_busy_percent || echo -1)"
                if [[ "$cpu_busy" =~ ^[0-9]+$ ]] &&
                   (( cpu_busy > MAX_CPU_BUSY_PERCENT )); then
                    reason+="CPU busy ${cpu_busy}% > ${MAX_CPU_BUSY_PERCENT}%; "
                fi
            fi

            if [[ -n "$reason" ]]; then
                breach_count=$((breach_count + 1))
                log "WATCHDOG breach ${breach_count}/${SAFETY_BREACH_SAMPLES}: $reason"

                if (( breach_count >= SAFETY_BREACH_SAMPLES )); then
                    {
                        echo "SAFETY STOP at $(ts)"
                        echo "$reason"
                    } > "$SAFETY_STOP_FILE"

                    log "ERROR: Sustained safety threshold violation. Stopping run."
                    terminate_command
                    exit 1
                fi
            else
                breach_count=0
            fi

            sleep "$WATCHDOG_INTERVAL"
        done
    ) >> "$STATUS_LOG" 2>&1 &
    WATCHDOG_PID=$!
}

write_final_snapshot() {
    {
        echo "===== $(ts) FINAL SNAPSHOT ====="
        echo
        echo "---- selected GPUs ----"
        local gpu
        for gpu in "${GPU_IDS[@]}"; do
            gpu="${gpu//[[:space:]]/}"
            nvidia-smi -i "$gpu" \
                --query-gpu=index,name,temperature.gpu,power.draw,power.limit,utilization.gpu,memory.used,memory.total \
                --format=csv || true
        done
        echo

        echo "---- system summary ----"
        echo "cpu_busy_percent=$(get_cpu_busy_percent || echo -1)"
        echo "load_per_cpu=$(get_load_per_cpu || echo -1)"
        echo "hottest_cpu_temp_c=$(get_hottest_cpu_temp || echo -1)"
        echo "available_mem_gib=$(get_available_mem_gib || echo -1)"
        echo "swap_used_percent=$(get_swap_used_percent || echo -1)"
        echo "free_disk_gib=$(get_free_disk_gib || echo -1)"
        echo

        sensors || true
        echo
        free -h || true
        echo
        df -hT "$WATCH_DISK_PATH" || true
        echo
        log_user_aggregates
        echo
        ps -eo user:18,pid,ppid,%cpu,%mem,rss,etime,stat,cmd \
            --sort=-rss | head -n 50 || true
        echo

        if [[ -f "$SAFETY_STOP_FILE" ]]; then
            echo "---- SAFETY STOP ----"
            cat "$SAFETY_STOP_FILE"
        fi
    } > "$FINAL_LOG" 2>&1
}

validate_numeric_settings
validate_gpu_ids
pin_cuda_devices_by_uuid

if [[ ! -d "$WATCH_DISK_PATH" ]]; then
    log "ERROR: WATCH_DISK_PATH does not exist: $WATCH_DISK_PATH"
    exit 2
fi

log "Run directory: $RUN_DIR"
write_env_snapshot "$@"
apply_power_cap
wait_for_safe_start
start_loggers

log "Running command with nice=$NICE_LEVEL and ionice class=$IONICE_CLASS level=$IONICE_LEVEL."
log "Command output: $CMD_LOG"

RUN_PREFIX=(nice -n "$NICE_LEVEL")
if command -v ionice >/dev/null 2>&1; then
    RUN_PREFIX=(ionice -c "$IONICE_CLASS" -n "$IONICE_LEVEL" "${RUN_PREFIX[@]}")
fi

setsid "${RUN_PREFIX[@]}" "$@" > >(tee "$CMD_LOG") 2>&1 &
COMMAND_PID=$!

start_safety_watchdog

set +e
wait "$COMMAND_PID"
RC=$?
set -e
COMMAND_PID=""

stop_pid "${WATCHDOG_PID:-}"
WATCHDOG_PID=""

if [[ -f "$SAFETY_STOP_FILE" ]]; then
    RC=75
    log "Run terminated by the safety watchdog."
fi

echo "$RC" > "$RUN_DIR/exit_code.txt"
write_final_snapshot

log "Command exit code: $RC"
log "Finished. Logs are in:"
log "  $RUN_DIR"

exit "$RC"
