#!/usr/bin/env python3 -u
from fairseq.models.roberta import XLMRModel
import torch, numpy
import torch.nn as nn
import argparse
import os,re


def main():
    parser = argparse.ArgumentParser(description='python code to extract target embedding from XLMR model.pt')
    # fmt: off
    parser.add_argument('--modeldir', help='path to XLMR model')
    parser.add_argument('--fseqdir', help='path to processed fairseq files')
    parser.add_argument('--outfile', help='path to output .pt files that contain target embedding')
    parser.add_argument('--src', help='the source language')
    parser.add_argument('--tgt', help='the target language')

    # fmt: on
    args = parser.parse_args()

    dict_xlmr = ['<bos>','<pad>','<eos>','<unk>'] 
    dict_en = ['<bos>','<pad>','<eos>','<unk>'] 
    index_en = [0, 1, 2, 3]
    print("now start to extract target embeddings ...")
    with open(args.fseqdir+'/dict.{}.txt'.format(args.src),'r') as fxlmr:
        for line in fxlmr:
            word, _ = line.strip('\n').split(' ')
            dict_xlmr.append(word) 

    count = 4 
    unkcount=0
    with open(args.fseqdir+'/dict.{}.txt'.format(args.tgt),'r') as ftgt:
        for line in ftgt:
            word, _ = line.strip('\n').split(' ')
            dict_en.append(word)
            if word in dict_xlmr:
                index = dict_xlmr.index(word)
            else:
                index = -10
                unkcount = unkcount + 1
            index_en.append(index)
            count = count + 1

    XLMR = XLMRModel.from_pretrained(args.modeldir, checkpoint_file='model.pt')
    xlmr_emb = XLMR.model.encoder.sentence_encoder.embed_tokens.weight  

    padding_idx = 1
    mx = nn.Embedding(count, 768, padding_idx=1) 
    nn.init.normal_(mx.weight, mean=0, std=768 ** -0.5)
    nn.init.constant_(mx.weight[padding_idx], 0)

    for idx in range(count):
        if index_en[idx] < 0:
            continue 
        mx.weight[idx] = xlmr_emb[index_en[idx]]
    torch.save(mx, args.outfile)

if __name__ == '__main__':
    main()