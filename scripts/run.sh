#!/bin/bash
train_1stage_NMT(){   
    workloc=$1 
    task=$2  
    src=de
    tgt=en
    modeldir=$workloc/models/${DATASET}_${task}_${MPLM} && mkdir -p $modeldir
    xlmr_modeldir=$workloc/models/${MPLM}_base

    enc_settings='--encoder-embed-dim 768 '
    dec_settings='--decoder-ffn-embed-dim 3072 --decoder-attention-heads 12 --decoder-layers 12 --decoder-embed-dim 768 '
    fseq=$workloc/fseq
    lr_rate=0.0005

    python train.py $fseq -a transformer_sixt --optimizer adam --lr ${lr_rate} -s $src -t $tgt \
        --label-smoothing 0.1 --dropout 0.3 --max-tokens 2048  --update-freq 2 --seed 16 --fp16  \
        --min-lr '1e-09' --lr-scheduler inverse_sqrt --weight-decay 0.01 --activation-fn gelu_accurate \
        --criterion label_smoothed_cross_entropy --max-update 200000  ${enc_settings} ${dec_settings}  \
        --warmup-updates 4000 --warmup-init-lr '1e-07' --keep-last-epochs 500 --save-interval-updates 5000 \
        --adam-betas '(0.9, 0.98)' --save-dir $modeldir --ddp-backend no_c10d --no-epoch-checkpoints \
        --max-source-positions 512  --xlmr-task $task --share-decoder-input-output-embed  \
        --xlmr-modeldir ${xlmr_modeldir} --tensorboard-logdir $modeldir/tensorboard \
        --eval-bleu --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
        --best-checkpoint-metric bleu --maximize-best-checkpoint-metric --log-format 'tqdm' \
        --clip-norm 2.0 --num-workers 0 --mplm-type ${MPLM}  \
        --eval-bleu-remove-bpe --eval-bleu-detok 'moses' --log-interval 100  \
    2>&1 | tee $modeldir/train.out
}

train_2stage_NMT(){   
    workloc=$1 
    task=$2 
    src=de
    tgt=en
    MaxUpdates=30000
    modeldir=$workloc/models/${DATASET}_${task}_${MPLM} && mkdir -p $modeldir
    xlmr_modeldir=$workloc/models/${MPLM}_base

    fseq=$workloc/fseq  
    enc_settings='--encoder-embed-dim 768 '
    dec_settings='--decoder-ffn-embed-dim 3072 --decoder-attention-heads 12 --decoder-layers 12 --decoder-embed-dim 768  '

    python train.py $fseq  -a transformer_sixt --optimizer adam --lr 0.0001 -s $src -t $tgt \
        --label-smoothing 0.1 --dropout 0.3 --max-tokens 1024 --update-freq 4 --seed 16 --fp16 \
        --min-lr '1e-09' --lr-scheduler inverse_sqrt --weight-decay 0.0 --activation-fn gelu_accurate \
        --criterion label_smoothed_cross_entropy --max-update 30000  --resdrop-layer 10 \
        --warmup-updates 10 --warmup-init-lr 0.00002 --keep-last-epochs 30 --no-epoch-checkpoints \
        --adam-betas '(0.9, 0.98)' --save-dir $modeldir --ddp-backend no_c10d \
        --share-decoder-input-output-embed --save-interval-updates 5000 ${enc_settings} ${dec_settings}  \
        --max-source-positions 512  --xlmr-task $task  --xlmr-modeldir ${xlmr_modeldir}  \
        --eval-bleu --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
        --best-checkpoint-metric bleu --maximize-best-checkpoint-metric --log-format 'tqdm' \
        --num-workers 0 --clip-norm 2.0 --tensorboard-logdir $modeldir/tensorboard \
        --fp16-no-flatten-grads --ft-last-f-dir $workloc/models/${DATASET}_encfix_embfix_decrand  \
        --mplm-type ${MPLM} --log-interval 100  \
    2>&1 | tee  $modeldir/train.out 

}

test_SixT_model(){ 
    workloc=$1 &&  task=$2 && src=$3  && tgt=en
    modeldir=$workloc/models/${DATASET}_${task}_${MPLM}
    xlmr_modeldir=$workloc/models/${MPLM}_base
    resdir=$modeldir/genres/${src}2${tgt} && mkdir -p $resdir
    suffix='_wmt'

    fseq=$workloc/fseq
    raw_reference=$workloc/raw/${src}2${tgt}/test.${tgt}

    python generate.py $fseq -s $src -t $tgt  --path $modeldir/checkpoint_best.pt \
        --max-tokens 4000 --beam 5 --sacrebleu \
        --remove-bpe  --decoding-path $resdir --mplm-type ${MPLM} --xlmr-task $task  > $resdir/gen_out

    cat $resdir/gen_out | grep -P "^H" | sort -V | cut -f 3-  > $resdir/decoding.txt
    python scripts/spm_decode.py --model ${xlmr_modeldir}/sentencepiece.bpe.model  \
        --input_format piece --input $resdir/decoding.txt > $resdir/decoding.detok
        
    echo "BLEU score for ${src}-2-en is ...."
    cat $resdir/decoding.detok | sacrebleu $raw_reference
}

export DATASET=wmt 
export MPLM=xlmr
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

## Here is an example to train SixT with XLM-R base, if you want to train with XLM-R large, you need to adjust the configs in the 'enc_settings' and 'dec_settings'.

workloc=/path/to/your/experiment/dir

echo ">> Now start to train the first stage NMT model with XLM-R"
task=encfix_embfix_decrand 
train_1stage_NMT $workloc $task

echo ">> Now start to fine-tune to get the second stage NMT model"
task=xlmr_2stage_posdrop  ## If you want to try the case where Resdrop is not used, set task='xlmr_2stage' and run.
train_2stage_NMT $workloc $task

echo ">> Now test the trained SixT model on De-En and Es-En testset"
for src in 'de' 'es';do 
    echo "start to translate $src - En testsets"
    test_SixT_model $workloc $task $src
done
