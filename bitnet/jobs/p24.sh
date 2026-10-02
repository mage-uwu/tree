# phase 2 rerun (degenerate-split fix) + phase 4 one-layer tree MLP sweep
python3 phase12.py --skip_baseline 1 --out /root/out/phase2.jsonl
python3 phase4.py --out /root/out/phase4.jsonl
