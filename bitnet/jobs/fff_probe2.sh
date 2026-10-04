# phase 17 probe 2 (after fixes: running coverage, SVD-initialised router, longer warm-up), two arms, 250 steps each
# (4.1M tokens): per-neuron control (B_leaf 1 = Phase-13 selector without rescore, now trained jointly) vs FFF 8-neuron leaves.
for B in 1 8; do
python3 fff_distill.py --B_leaf $B --k 1024 --k_start 1536 --anneal_frac 0.3 --steps 250 --mb 4 --accum 4 --T 1024 \
  --eval_every 125 --save_every 250 --save /root/out/fff_probe2_B$B.pt --out /root/out/fff_probe2_B$B.jsonl
done
