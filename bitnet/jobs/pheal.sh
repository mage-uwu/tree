# phase 12: healing. All 30 MLPs replaced by students bootstrapped from teacher neurons, end-to-end KL training.
python3 phase_heal.py --K 1 --m 1024 --out /root/out/phase_heal.jsonl
python3 phase_heal.py --K 8 --m 512  --out /root/out/phase_heal.jsonl
python3 phase_heal.py --K 1 --m 512  --out /root/out/phase_heal.jsonl
python3 phase_heal.py --K 1 --m 1024 --ternary 1 --out /root/out/phase_heal.jsonl
