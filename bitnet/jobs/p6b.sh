# all three conversions together (rerun after pod loss)
python3 phase6.py --tau_sel 8 --mlp_frac 0.99 --combos attn+mlp+head --out /root/out/phase6.jsonl
python3 phase6.py --tau_sel 6 --mlp_frac 0.98 --combos attn,attn+mlp+head --out /root/out/phase6.jsonl
