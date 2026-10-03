# decode speed vs neurons per token k (selector path, no rescore; quality ignored: timing only)
cd /tmp/claude-0/eng/BitNet
B="build/bin/llama-completion -n 256 -c 9000 --temp 0 --ignore-eos -no-cnv"
HEAD="TREE_HEAD=../vocab_trees_S128.bin TREE_HEAD_N=8192"
ATT="TREE_ATTN=../key_trees_S32.bin TREE_TAU=4 TREE_KV8=1"
run() { echo "== $1"; shift; env "$@" 2>&1 | grep -E "eval time" | grep -v prompt | sed 's/.*eval time/eval/'; }
for setup in "short1:-t 1 -f ../prompt_short.txt -fa off" "short4:-t 4 -f ../prompt_short.txt -fa off" "long4:-t 4 -f ../prompt8k.txt -fa on"; do
  name=${setup%%:*}; args=${setup#*:}; X=""; [ $name = long4 ] && X="$ATT"
  echo "######## $name"
  run "stock" $B $args -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf $( [ $name = long4 ] && echo "-fa off" )
  run "head + sparse0.99" $HEAD TREE_MLP_FRAC=0.99 $X $B $args -m ../hfcodes.gguf
  for k in 1536 1024 768 512 256 16; do
    run "head + sel k=$k" $HEAD TREE_MLP_SEL=../sel_trained.bin TREE_MLP_K=$k TREE_MLP_SELC=$k $X $B $args -m ../healed_nr.gguf
  done
done
