#!/usr/bin/env bash
# Run the correctness test with mpirun.
#
#   tests/run_tests.sh [path/to/test_correctness]
#
# NPROCS (default "4") lists the process counts to try, e.g. NPROCS="2 3 4 5".
# --oversubscribe is added for Open MPI; --allow-run-as-root when run as root.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
exe="$root/build/test_correctness"
if [ $# -gt 0 ]; then
    exe="$1"
    shift
fi
nprocs="${NPROCS:-4}"

if [ ! -x "$exe" ]; then
    echo "building into $root/build"
    cmake -S "$root" -B "$root/build" -DCMAKE_BUILD_TYPE=Release >/dev/null
    cmake --build "$root/build" --target test_correctness >/dev/null
    exe="$root/build/test_correctness"
fi

flags=()
if mpirun --version 2>&1 | grep -q "Open MPI"; then
    flags+=(--oversubscribe)
    if [ "$(id -u)" -eq 0 ]; then
        flags+=(--allow-run-as-root)
    fi
fi

status=0
for np in $nprocs; do
    echo "== mpirun -np $np"
    if ! mpirun -np "$np" "${flags[@]}" "$exe" "$@"; then
        status=1
    fi
done

if [ "$status" -eq 0 ]; then
    echo "all runs passed"
else
    echo "some runs failed"
fi
exit "$status"
