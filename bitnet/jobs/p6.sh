# combine attention select+rescore, sparse exact MLP, tree output layer: each alone and together
python3 phase6.py --tau_sel 8 --mlp_frac 0.99 --out /root/out/phase6.jsonl
python3 phase6.py --tau_sel 6 --mlp_frac 0.98 --combos attn,mlp,attn+mlp+head --out /root/out/phase6.jsonl
