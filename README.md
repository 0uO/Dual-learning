# Dual-learning for machine translation
-------

An implementation of [Dual Learning For Machine Translation](https://arxiv.org/abs/1611.00179) and [Joint Training for Neural Machine Translation Models with Monolingual Data](https://arxiv.org/abs/1803.00353) on tensorflow.

INSTALLATION
------------
This project depend heavily on [nematus v0.3. ]( https://github.com/EdinburghNLP/nematus )

Nematus requires the following packages:

 - Python >= 2.7
 - tensorflow

See more details about nematus in above link.

And I use kenlm as language model:

- Kenlm ( pip install [https://github.com/kpu/kenlm/archive/master.zip](https://github.com/kpu/kenlm/archive/master.zip) )

It seems you need complie it from source code for getting binary executing file. See more details about kenlm in the link above.

The code inside which related to language model are independent, so you could use other language model as long as it could offer the function of ***score a sentence*** . 
 
USAGE INSTRUCTIONS
------------------

You shall prepare the following models:

- A pair of small parallel dataset of two language.
- A pair of large monolingual dataset of two language.
- NMT model X 2, using nematus and small dataset.
- Language model X2 , using the script of /LM/train_lm.py. You need set the KENLM_PATH and TEMP_DIR inside.

I preprocessed dataset by [subword](https://github.com/rsennrich/subword-nmt).

Then set the parameter in /test/test_train_dual.sh , especial :
- LdataPath
- SdataPath
- modelDir
- LMDir

Description as their name. And you could write your own training script, see the following new added configs for dual learning:

#### dual; parameters for dual learning.
| parameter | description |
|---        |---          |
| --dual | active dual learning |
| --para | active parallel dataset using in dual learning |
| --reinforce | active reinforcement learning |
| --alpha|weight of lm score in dual learning. |
| --joint | active joint training |
| --model_rev or --saveto_rev | reverse model file name |
| --reload_rev | load existing model from this path. Set to \"latest_checkpoint\" to reload the latest checkpoint in the same directory of --model |
| --source_lm | language model (source) |
| --target_lm | language model (target) |
| --lms | language models (one for source, and one for target.) |
| --source_dataset_mono | parallel training corpus (source) |
| --target_dataset_mono | parallel training corpus (target) |
| --datasets_mono | parallel training corpus (one for source, and one for target.) | 

For replaying the paper of [Dual Learning For Machine Translation](https://arxiv.org/abs/1611.00179), you need add  --reinforce.
For replaying the paper [Joint Training for Neural Machine Translation Models with Monolingual Data](https://arxiv.org/abs/1611.00179), you need add  --reinforce.

#### TODO

The result of dual learning isn't good, while joint training works well, later I would post the results.

#### History

##### V1.0

- basically replay the paper of [Dual Learning For Machine Translation](https://arxiv.org/abs/1611.00179).

##### V1.1

- bug fixed.

##### V1.2

- bug fixed and improve the info displaying.
- basically replay the paper of [Joint Training for Neural Machine Translation Models with Monolingual Data](https://arxiv.org/abs/1611.00179).

##### V1.3

- bug fixed.
