# hybrid partial-sum + residual trees vs partial-sum alone, per layer and all layers
python3 phase_treegate.py --layers 15,2,28 --trees "" --partial "" --hybrid 128:32:2048,256:32:2048,256:32:1536,128:64:2048 --all "" --all_partial 256:3072,512:2048 --all_hybrid 256:32:2048,128:32:2048 --out /root/out/phase_hybrid.jsonl
