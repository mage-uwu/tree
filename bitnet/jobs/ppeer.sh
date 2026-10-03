# phase 13: teacher-supervised neuron selection (svd / low-rank / PEER product keys) + select-then-rescore, then
# all 30 layers and end-to-end healing of the selected neurons.
python3 phase_peer.py --out /root/out/phase_peer.jsonl --stageB_selector lr --stageB_rescore 1
