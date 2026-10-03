# phase 13b: ternary-constrained healing (engine-deployable) of low-rank+rescore selected neurons, k=1536 and k=1024
python3 phase_peer.py --out /root/out/phase_peer2.jsonl --stageA_layers "" --stageB_selector lr --stageB_rescore 1 --stageB_ks 1536 --heal_k 1536 --heal_ternary 1 --heal_steps 3000 --heal_lr 3e-4 --eval_every 500
python3 phase_peer.py --out /root/out/phase_peer2.jsonl --stageA_layers "" --stageB_selector lr --stageB_rescore 1 --stageB_ks 1024 --heal_k 1024 --heal_ternary 1 --heal_steps 3000 --heal_lr 3e-4 --eval_every 500
