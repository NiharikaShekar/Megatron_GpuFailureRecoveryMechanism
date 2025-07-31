#!/bin/bash

i=0
MAX_JOBS=4
for file in ../raw_data/text/*/*; do
    if [ -f "$file" ]; then
        output="my-gpt_${i}"
        echo "Preproc $file -> saving as $output"
        python ../../../megatron/tools/preprocess_data.py \
        --input $file \
        --workers 1 \
        --output-prefix $output \
        --vocab-file ../vocab_merge_files/gpt2-vocab.json \
        --tokenizer-type GPT2BPETokenizer \
        --merge-file ../vocab_merge_files/gpt2-merges.txt \
        --append-eod &
        i=$((i+1))
        while [ "$(jobs -r | wc -l)" -ge "$MAX_JOBS" ]; do
            sleep 1
        done
    fi
done