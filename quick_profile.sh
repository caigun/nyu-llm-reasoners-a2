#!/bin/bash
# Quick profiling test script

echo "========================================="
echo "Profiling Forward Pass"
echo "========================================="
uv run nsys profile --trace cuda,nvtx --stats=true --output=nsight_log/profile_both --force-overwrite=true \
  python student/benchmark.py --profile both --seq_length 128 --n_steps 100