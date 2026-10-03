# phase 13d: trained selectors WITH rescore (kc = 2k candidates, exact gate picks k=1536), ternary healing 10M tokens, saved
python3 phase_peer.py --out /root/out/phase_peer4.jsonl --stageA_layers "" --stageB_selector lr --stageB_rescore 1 --stageB_ks 1536 --heal_k 1536 --heal_ternary 1 --heal_steps 2500 --heal_lr 3e-4 --eval_every 500 --save /root/out/peer_k1536r
