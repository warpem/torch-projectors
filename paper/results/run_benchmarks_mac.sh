python paper/results/benchmarks/forward_2d.py --device mps --platform m4-mps --warmup-runs 3 --timing-runs 15 --test-cooldown 10 --title "M4: MPS"
sleep 180
python paper/results/benchmarks/forward_2d.py --device cpu --platform m4-cpu --warmup-runs 3 --timing-runs 15 --test-cooldown 10 --title "M4: CPU"
sleep 180
python paper/results/benchmarks/backward_2d.py --device mps --platform m4-mps --warmup-runs 3 --timing-runs 15 --test-cooldown 10 --title "M4: MPS"
sleep 180
python paper/results/benchmarks/backward_2d.py --device cpu --platform m4-cpu --warmup-runs 3 --timing-runs 15 --test-cooldown 10 --title "M4: CPU"
sleep 180
python paper/results/benchmarks/forward_3d_to_2d.py --device mps --platform m4-mps --warmup-runs 3 --timing-runs 15 --test-cooldown 10 --title "M4: MPS"
sleep 180
python paper/results/benchmarks/forward_3d_to_2d.py --device cpu --platform m4-cpu --warmup-runs 3 --timing-runs 15 --test-cooldown 10 --title "M4: CPU"
sleep 180
python paper/results/benchmarks/backward_2d_to_3d.py --device mps --platform m4-mps --warmup-runs 3 --timing-runs 15 --test-cooldown 10 --title "M4: MPS"
sleep 180
python paper/results/benchmarks/backward_2d_to_3d.py --device cpu --platform m4-cpu --warmup-runs 3 --timing-runs 15 --test-cooldown 10 --title "M4: CPU"
sleep 180
python paper/results/benchmarks/forward_3d_to_2d_comparison.py --device cpu --platform m4-cpu --warmup-runs 3 --timing-runs 15 --test-cooldown 10 --title "M4: CPU" --interpolations linear
sleep 180
python paper/results/benchmarks/backward_2d_to_3d_comparison.py --device cpu --platform m4-cpu --warmup-runs 3 --timing-runs 15 --test-cooldown 10 --title "M4: CPU" --interpolations linear --disable-tfs-backward-safety