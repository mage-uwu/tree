# sparse exact MLP, round 2: floor + weighted gate + tree leaf sets at layer 15; full comparison at layers 2, 28
python3 phase4b.py --layers 15 --sels floor,gatew,tree --out /root/out/phase4b.jsonl
python3 phase4b.py --layers 2,28 --sels floor,oracle,gate,gatew,tree --out /root/out/phase4b.jsonl
