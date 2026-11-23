#!/usr/bin/env bash
# Sets up the Clinical Interview LLM Pipeline in this folder.
#
#   bash install.sh                     the cilp command
#   bash install.sh train               plus LoRA training and prediction
#   bash install.sh asr diarize train   plus several optional parts (or: all)
#
# Optional parts: asr (faster-whisper transcription), diarize (pyannote
# diarization), train (LoRA training and prediction on an NVIDIA GPU).
#
# The script uses an installed Python 3.11, 3.12, or 3.13. Without one, it
# downloads uv and, with it, Python 3.12 into .tools in this folder. It creates
# the environment .venv, installs the pipeline, runs the tests, and shows
# cilp --help. Running it again keeps .venv and adds the parts named.

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

set -euo pipefail

UV_VERSION="0.13.7"
UV_PYTHON="3.12"

ROOT="$(cd "$(dirname "$0")" && pwd -P)"
SCRIPT="$ROOT/$(basename "$0")"
VENV="$ROOT/.venv"
TOOLS="$ROOT/.tools"
SYSTEM="$(uname -s)"
cd "$ROOT"

say() { printf '%s\n' "$*"; }
step() { printf '\n==> %s\n' "$*"; }
fail() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }
usage() { sed -n '2,14p' "$SCRIPT" | sed 's/^# \{0,1\}//'; }

EXTRAS=""
add_extra() {
    case ",$EXTRAS," in
        *",$1,"*) ;;
        *) EXTRAS="${EXTRAS:+$EXTRAS,}$1" ;;
    esac
}
for arg in "$@"; do
    case "$arg" in
        asr|diarize|train) add_extra "$arg" ;;
        all) add_extra asr; add_extra diarize; add_extra train ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; fail "Unknown part '$arg'. Choose from asr, diarize, train, all." ;;
    esac
done

case "$SYSTEM" in
    Linux|Darwin) ;;
    *) fail "This script runs on macOS and Linux. On Windows, install WSL (wsl --install in PowerShell) and run it in the Ubuntu terminal." ;;
esac

# True if the given interpreter is Python 3.11 to 3.13, the versions the tests cover.
python_ok() {
    "$1" -c 'import sys; sys.exit(0 if (3, 11) <= sys.version_info[:2] <= (3, 13) else 1)' \
        >/dev/null 2>&1
}

find_python() {
    local name path
    for name in python3.12 python3.13 python3.11 python3 python; do
        path="$(command -v "$name" 2>/dev/null)" || continue
        # Apple's /usr/bin/python3 is 3.9 and may open an installation dialog.
        if [ "$SYSTEM" = Darwin ] && [ "$path" = /usr/bin/python3 ]; then
            continue
        fi
        if python_ok "$path"; then
            printf '%s\n' "$path"
            return 0
        fi
    done
    return 1
}

download() {
    if command -v curl >/dev/null 2>&1; then
        curl --proto '=https' --tlsv1.2 -fsSL "$1" -o "$2"
    elif command -v wget >/dev/null 2>&1; then
        wget -q --https-only -O "$2" "$1"
    else
        fail "Downloading needs curl or wget. Install one of them and run the script again."
    fi
}

find_uv() {
    if command -v uv >/dev/null 2>&1; then
        command -v uv
        return 0
    fi
    local candidate
    for candidate in "$TOOLS/uv/uv" "$TOOLS/uv/bin/uv"; do
        if [ -x "$candidate" ]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    return 1
}

install_uv() {
    say "Downloading uv $UV_VERSION into $TOOLS/uv"
    mkdir -p "$TOOLS"
    local installer="$TOOLS/uv-installer.sh"
    download "https://github.com/astral-sh/uv/releases/download/$UV_VERSION/uv-installer.sh" \
        "$installer" || fail "uv could not be downloaded. Check the internet connection."
    UV_UNMANAGED_INSTALL="$TOOLS/uv" UV_PRINT_QUIET=1 sh "$installer" \
        || fail "uv could not be installed."
    rm -f "$installer"
}

venv_with_uv() {
    local uv
    if ! uv="$(find_uv)"; then
        install_uv
        uv="$(find_uv)" || fail "uv was installed, but its program was not found in $TOOLS/uv."
    fi
    say "Getting Python $UV_PYTHON through uv"
    UV_PYTHON_INSTALL_DIR="$TOOLS/python" UV_CACHE_DIR="$TOOLS/cache" \
        "$uv" venv --quiet --seed --managed-python --python "$UV_PYTHON" "$VENV" \
        || fail "uv could not create the environment. Check the internet connection."
}

step "Preparing Python"
if [ -e "$VENV" ]; then
    if [ -x "$VENV/bin/python" ] && python_ok "$VENV/bin/python" \
        && "$VENV/bin/python" -m pip --version >/dev/null 2>&1; then
        say "Using the existing environment $VENV"
    else
        fail "The folder .venv holds no usable Python 3.11 to 3.13. Remove it with
    rm -rf \"$VENV\"
and run bash install.sh again."
    fi
elif PYTHON="$(find_python)"; then
    say "Using $PYTHON ($("$PYTHON" -c 'import platform; print(platform.python_version())'))"
    if ! "$PYTHON" -m venv "$VENV" >/dev/null 2>&1; then
        # Some Linux systems ship Python without the venv module.
        rm -rf "$VENV"
        say "This Python cannot create environments, so uv provides one."
        venv_with_uv
    fi
else
    say "No Python 3.11 to 3.13 was found, so uv provides one."
    venv_with_uv
fi
PY="$VENV/bin/python"

step "Installing the pipeline${EXTRAS:+ with $EXTRAS}"
"$PY" -m pip install --quiet --disable-pip-version-check --upgrade pip \
    || fail "pip could not be updated. Check the internet connection."
"$PY" -m pip install --disable-pip-version-check -e ".${EXTRAS:+[$EXTRAS]}" \
    || fail "The installation stopped. The messages above show the package that failed."

step "Running the tests"
if ! tests="$("$PY" -m unittest discover -s tests 2>&1)"; then
    # The test report starts at the first ====== line. Command output fills the rest.
    report="$(printf '%s\n' "$tests" | awk '/^======/ { show = 1 } show { print } /^FAILED/ { exit }')"
    if [ -n "$report" ]; then
        printf '%s\n' "$report"
    else
        printf '%s\n' "$tests" | tail -n 40
    fi
    fail "Some tests failed. The report above names them."
fi
printf '%s\n' "$tests" | grep -E '^(Ran [0-9]+ tests|OK)' || true

step "Checking the cilp command"
"$VENV/bin/cilp" --help

case ",$EXTRAS," in
    *",train,"*)
        say ""
        if [ "$SYSTEM" = Darwin ]; then
            say "Note: training and prediction need Linux with an NVIDIA GPU. On this Mac, the train part gives cilp plan its length check."
        elif ! command -v nvidia-smi >/dev/null 2>&1; then
            say "Note: no NVIDIA driver was found (nvidia-smi). Training and prediction need an NVIDIA GPU."
        elif gpus="$("$PY" -c 'import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)' 2>/dev/null)" \
            && [ "${gpus:-0}" -gt 0 ]; then
            say "PyTorch sees $gpus GPU(s) for training and prediction."
        else
            say "Note: PyTorch cannot reach the GPU. Its CUDA version may need a newer NVIDIA driver than the one nvidia-smi shows. The PyTorch website lists builds for older drivers."
        fi
        ;;
esac
case ",$EXTRAS," in
    *",diarize,"*)
        if ! command -v ffmpeg >/dev/null 2>&1; then
            say ""
            say "Note: diarization reads audio through FFmpeg. Install it with 'brew install ffmpeg' (macOS with Homebrew) or 'sudo apt install ffmpeg' (Ubuntu)."
        fi
        ;;
esac

cat <<EOF

Done. Clinical Interview LLM Pipeline $("$VENV/bin/cilp" --version) is installed in $VENV

In every new terminal, activate the environment once:

    source "$VENV/bin/activate"

Then the cilp commands work in that terminal, for example:

    cilp --help
EOF
