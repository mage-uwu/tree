# phase 17 probe 3: per-neuron control of probe 2 (B_leaf 1), micro-batch halved (router tensors are 8x larger), 250 steps.
python3 fff_distill.py --B_leaf 1 --k 1024 --k_start 1536 --anneal_frac 0.3 --steps 250 --mb 2 --accum 8 --T 1024 \
  --eval_every 125 --save_every 250 --save /root/out/fff_probe2_B1.pt --out /root/out/fff_probe2_B1.jsonl
