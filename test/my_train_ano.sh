# warning: this test is useful to check if training fails, and what speed you can achieve
# the toy datasets are too small to obtain useful translation results,
# and hyperparameters are chosen for speed, not for quality.
# For a setup that preprocesses and trains a larger data set,
# check https://github.com/rsennrich/wmt16-scripts/tree/master/sample

LdataPath=/data2/Europarl_1x2/large
SdataPath=/data2/Europarl_1x2/small

L1=de
L2=en
modelDir=/home/lyy/myDualLearning/models
LMDir=/home/lyy/myDualLearning/LM

VS=/home/lyy/myDualLearning/transOut/test_bleu_${L2}.py

python2 -u ../nematus/nmt.py \
  --model ${modelDir}/${L1}2${L2}/model.${L1}2${L2}.npz \
  --datasets ${SdataPath}/corpus.bpe.${L1} ${SdataPath}/corpus.bpe.${L2} \
  --valid_source_dataset ${LdataPath}/smaller/corpus.bpe.${L1} \
  --valid_target_dataset ${LdataPath}/smaller/corpus.bpe.${L2} \
  --dictionaries ${LdataPath}/corpus.bpe.${L1}.json ${LdataPath}/corpus.bpe.${L2}.json \
  --dim_word 256 \
  --dim 512 \
  --maxlen 100 \
  --optimizer adam \
  --lrate 0.0001 \
  --batch_size 50 \
  --valid_batch_size 50\
  --validFreq 1000\
  --no_shuffle \
  --dispFreq 5000 \
  --sampleFreq 5000 \
  --beamFreq 10000 \
  --saveFreq 20000 \
  --max_epochs 200 \
  --n_words_src 60000 \
  --n_words 60000 \
  --reload latest_checkpoint \
  --patience 100 \
  # --reload latest_checkpoint \
  # --no_reload_training_progress \
  # --valid_datasets $dataPath1/newsdev2016.bpe.${L1} $dataPath1/newsdev2016.bpe.${L2} \
