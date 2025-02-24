#bin/bash

for file in examples/model_outputs/*.json; do
    echo $file
    python3 norm.py benchmark $file --to_leaderboard
done