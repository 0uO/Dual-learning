# Dual-learning for machine translation
-------

An implementation of [Dual Learning For Machine Translation](https://arxiv.org/abs/1611.00179) on tensorflow.

INSTALLATION
------------
This project is heavily depend on [nematus v0.3. ]( https://github.com/EdinburghNLP/nematus )

Nematus requires the following packages:

 - Python >= 2.7
 - tensorflow

See more details about nematus in above link.

And I use kenlm as language model:

- Kenlm ( pip install [https://github.com/kpu/kenlm/archive/master.zip](https://github.com/kpu/kenlm/archive/master.zip) )

It seems you need complie it from source code for getting binary executing file. See more details about kenlm in above link.

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
| --reinforce | active reinforcement learning |
| --para | active parallel dataset using in dual learning |
| --model_rev or --saveto_rev | reverse model file name |
| --source_lm | language model (source) |
| --target_lm | language model (target) |
| --lms | language models (one for source, and one for target.) |
| --source_dataset_mono | parallel training corpus (source) |
| --target_dataset_mono | parallel training corpus (target) |
| --datasets_mono | parallel training corpus (one for source, and one for target.) |
| --dual |weight of lm score. |

For replaying the paper, you need add  --reinforce , else it would be a normal back-translation method.

#### TODO

The result isn't good.