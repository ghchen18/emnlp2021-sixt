# EMNLP'21 Zero-shot Cross-lingual Transfer of NMT with Multilingual Pretrained Encoders 

This is the official repo for the EMNLP 2021 paper "Zero-shot Cross-lingual Transfer of Neural Machine Translation with Multilingual Pretrained Encoders" [[paper](https://aclanthology.org/2021.emnlp-main.2/)].

The code uses pytorch and is based on [`fairseq` v0.9.0](https://github.com/pytorch/fairseq/tree/v0.9.0). Please refer to the official `fairseq` repo for more details about the instructions and examples about `fairseq` toolkit. You can follow the steps below to replicate the experiments in the paper. Before start, first install relevant packages. The `fairseq` package should be built with this repo.

```bash
## create conda env first
cd /path/to/this/repo
python -m pip install -e . --user   ## install fairseq toolkit with this repo
```

## Step 1: preprocess data

First download the off-the-shelf xlmr-base and xlmr-large models from the [official repo](https://github.com/pytorch/fairseq/blob/main/examples/xlmr/README.md#pre-trained-models). Then download the training/validation/test data from WMT/WAT/CC-align/FLores/Tatoba etc., the detailed urls are in the paper appendix. Suppose all files are placed under a path `dataloc=/path/to/your/raw/data`. Then use the `fairseq` preprocess.py to binarize the dataset: `bash scripts/preprocess.sh`, more details are in the shell script. 


## Step 2: Train the SixT model with two-stage training

The proposed SixT model can be trained with the processed dataset. `bash scripts/run.sh` to train SixT model in two training stages. More details are in the shell script.

## Step 3: Test the SixT model in the testsets

After the model is trained, you can directly test it in a zero-shot manner. The testsets are also needed to be binarized with the `preprocess.sh` script. See more in `scripts/run.sh`. 


## Citation

If you find our work useful for your research, welcome to cite the following paper:

```bibtex
@inproceedings{chen2021zeroshot,
    title={Zero-shot Cross-lingual Transfer of Neural Machine Translation with Multilingual Pretrained Encoders}, 
    author = "Chen, Guanhua  and
      Ma, Shuming  and
      Chen, Yun  and
      Dong, Li  and
      Zhang, Dongdong  and
      Pan, Jia  and
      Wang, Wenping  and
      Wei, Furu",
    year={2021},
    booktitle = "Proceedings of EMNLP",
    month = nov,
    year = "2021",
    pages = "15--26",
}

```
