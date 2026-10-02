# phase 12b: healing smaller cuts (must beat the lossless exact-gate sparse MLP, 1.9x fewer MACs). 12M tokens per arm.
python3 phase_heal.py --K 1 --m 2048 --steps 3000 --eval_every 500 --out /root/out/phase_heal2.jsonl
python3 phase_heal.py --K 4 --m 2048 --steps 3000 --eval_every 500 --out /root/out/phase_heal2.jsonl
python3 phase_heal.py --K 1 --m 3072 --steps 3000 --eval_every 500 --out /root/out/phase_heal2.jsonl
