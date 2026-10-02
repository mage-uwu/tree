# Engine notes

Baseline is stock bitnet.cpp (github.com/microsoft/BitNet, commit 0b341e5), built with
`python setup_env.py -md models/BitNet-b1.58-2B-4T -q i2_s` from the official `microsoft/BitNet-b1.58-2B-4T-gguf`.

`headbench.c` times the tied output layer (128256 x 2560) matvec inside ggml:

    gcc -O2 -I$BITNET/3rdparty/llama.cpp/ggml/include headbench.c -L$BITNET/build/bin \
        -lggml -lggml-base -lggml-cpu -Wl,-rpath,$BITNET/build/bin -o headbench
