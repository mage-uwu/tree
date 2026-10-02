#!/bin/sh
# End-to-end smoke test (~3-5 min on 2 CPU cores). Every step should print and none should error.
set -e
cd "$(dirname "$0")"
./build.sh
echo "== 1. base BitNet model: PyTorch vs C (reference path)"
python3 export_tree.py runs/full_shakespeare.pt runs/full_shakespeare.tre
EXACT=1 python3 parity_tree.py runs/full_shakespeare.pt runs/full_shakespeare.tre 128
echo "== 2. shipped tree attention (oblique 24x4, int8): parity, then fast path vs reference path"
python3 export_tree.py runs/full_tree_k_24x4_q8.pt runs/full_tree_k_24x4_q8.tre
EXACT=1 python3 parity_tree.py runs/full_tree_k_24x4_q8.pt runs/full_tree_k_24x4_q8.tre 128
python3 evalc.py runs/full_tree_k_24x4_q8.tre 8
echo "== 3. fresh attention conversion (tiny: 8 trees, 32 calibration sequences, no tuning)"
python3 convert.py --base runs/full_shakespeare.pt --configs 8x4 --values 0 --calib_seqs 32 --ft_steps 0 --save "8x4:runs/smoke_k_8x4.pt" --out runs/smoke_k.json
echo "== 4. fresh tree-MLP conversion (tiny: 64 leaves, rank 8), then int8"
python3 convert_mlp.py --base runs/full_shakespeare.pt --configs 6:0:8:0 --calib_seqs 128 --savedir runs --out runs/smoke_mlp.json
python3 quant_eval.py runs/full_mlptree_6_0_8_0.pt
echo "== 5. plug the two halves together, export, parity, benchmark"
python3 plug.py --attn runs/full_tree_k_24x4_q8.pt --mlp runs/full_mlptree_6_0_8_0.pt --save runs/smoke_all.pt
python3 export_tree.py runs/smoke_all.pt runs/smoke_all.tre
EXACT=1 python3 parity_tree.py runs/smoke_all.pt runs/smoke_all.tre 128
./treeattn runs/smoke_all.tre bench 2048
./scanbench 128 32
echo "== smoke test finished"
