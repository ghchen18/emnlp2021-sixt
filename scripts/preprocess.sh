#!/bin/bash
process_data(){
    src=de 
    tgt=en
    workloc=/path/to/your/work/location  ## suppose raw data is located in $workloc/raw 
    xlmr_model=/path/to/pretrained-xlmr-model  ## The off-the-shelf xlmr-base and xlmr-large can be downloaded from [this repo](https://github.com/pytorch/fairseq/blob/main/examples/xlmr/README.md#pre-trained-models)

    raw=$workloc/raw 
    bpe=$workloc/bpe
    fseq=$workloc/fseq && mkdir -p $raw $bpe $fseq 

    for L in $src $tgt; do
        for f in valid.$L test.$L train.$L; do 
            echo "apply sentencepiece to ${f}..."
            python scripts/spm_encode.py --model ${xlmr_model}/sentencepiece.bpe.model \
                --inputs $raw/$f --outputs $bpe/$f 
        done
    done

    python preprocess.py -s $src -t $tgt --dataset-impl lazy \
        --workers 16 --destdir $fseq --xlmr-task vanilla \
        --validpref $bpe/valid --testpref $bpe/test  \
        --trainpref $bpe/train --srcdict ${xlmr_model}/dict.txt 
    
    ## extract the decoder embeddings correspond to target vocabulary, i.e. decoder embedding is initialized with XLM-R
    python scripts/save_embed.py --fseqdir $fseq \
        --modeldir ${xlmr_model} --src $src --tgt $tgt \
        --outfile $fseq/xlmr_${tgt}_emb.pt

}

export DATASET=wmt
export CUDA_VISIBLE_DEVICES=0


## An example to show how to build binarized dataset from raw text
process_data
