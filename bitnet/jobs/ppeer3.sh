# phase 13c: trained selectors WITHOUT rescore (engine speed point: k=1536, C=k), saved first, then ternary healing, saved
python3 phase_peer.py --out /root/out/phase_peer3.jsonl --stageA_layers "" --stageB_selector lr --stageB_rescore 0 --stageB_ks 1536,2048 --heal_k 1536 --heal_ternary 1 --heal_steps 2500 --heal_lr 3e-4 --eval_every 500 --save /root/out/peer_k1536
