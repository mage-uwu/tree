# attention experiments (floor, TAU value pruning, one-layer select+rescore) + all-layer key conversion
python3 phase2b.py --out /root/out/phase2b.jsonl
python3 phase3.py --configs 32 --tau_sel 8 --out /root/out/phase3.jsonl
python3 phase3.py --configs 64,96 --out /root/out/phase3.jsonl --save /root/out
