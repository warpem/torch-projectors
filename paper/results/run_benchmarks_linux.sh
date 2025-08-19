python paper/results/benchmarks/forward_2d.py --device cuda --platform a100-cuda --warmup-runs 3 --timing-runs 15 --test-cooldown 10 --title "A100: CUDA"
sleep 180
python paper/results/benchmarks/backward_2d.py --device cuda --platform a100-cuda --warmup-runs 3 --timing-runs 15 --test-cooldown 10 --title "A100: CUDA"
sleep 180
python paper/results/benchmarks/forward_3d_to_2d.py --device cuda --platform a100-cuda --warmup-runs 3 --timing-runs 15 --test-cooldown 10 --title "A100: CUDA"
sleep 180
python paper/results/benchmarks/backward_2d_to_3d.py --device cuda --platform a100-cuda --warmup-runs 3 --timing-runs 15 --test-cooldown 10 --title "A100: CUDA"
sleep 180
python paper/results/benchmarks/forward_3d_to_2d_comparison.py --device cuda --platform a100-cuda --warmup-runs 3 --timing-runs 15 --test-cooldown 10 --title "A100: CUDA" --interpolations linear
sleep 180
python paper/results/benchmarks/backward_2d_to_3d_comparison.py --device cuda --platform a100-cuda --warmup-runs 3 --timing-runs 15 --test-cooldown 10 --title "A100: CUDA" --interpolations linear --disable-tfs-backward-safety