# phase 17 probe: FFF distillation fast path, all 30 layers, k 1536 -> 1024 (8-neuron leaves, low-rank flat router),
# 400 steps x 16k tokens (6.5M tokens): measures tokens/s + GPU memory, first part of the KL curve, saves a checkpoint.
python3 fff_distill.py --k 1024 --k_start 1536 --anneal_frac 0.3 --steps 400 --mb 4 --accum 4 --T 1024 \
  --eval_every 100 --save_every 400 --save /root/out/fff_probe.pt --out /root/out/fff_probe.jsonl
