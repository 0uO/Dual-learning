# warning: this test is useful to check if training fails, and what speed you can achieve
# the toy datasets are too small to obtain useful translation results,
# and hyperparameters are chosen for speed, not for quality.
# For a setup that preprocesses and trains a larger data set,
# check https://github.com/rsennrich/wmt16-scripts/tree/master/sample

LdataPath=/data2/Europarl_1x2/large
SdataPath=/data2/Europarl_1x2/small

L1=en
L2=de
# modelDir=/home/lyy/myDualLearning/models
modelDir=/data2/models/model_Lyy
modelDirSuffix=
LMDir=/home/lyy/myDualLearning/LM

python2 -u ../nematus/nmt.py \
  --dual \
  --joint \
  --para \
  --model ${modelDir}/${L1}2${L2}${modelDirSuffix}/model.${L1}2${L2}.npz \
  --model_rev ${modelDir}/${L2}2${L1}${modelDirSuffix}/model.${L2}2${L1}.npz \
  --lms ${LMDir}/${L1}.zip ${LMDir}/${L2}.zip\
  --datasets ${SdataPath}/corpus.bpe.${L1} ${SdataPath}/corpus.bpe.${L2} \
  --datasets_mono ${LdataPath}/corpus.bpe.${L1} ${LdataPath}/corpus.bpe.${L2} \
  --dictionaries ${SdataPath}/corpus.bpe.${L1}.json ${SdataPath}/corpus.bpe.${L2}.json \
  --dim_word 256 \
  --dim 512 \
  --maxlen 100 \
  --optimizer adam \
  --lrate 0.0001 \
  --batch_size 6 \
  --beam_size 8 \
  --no_shuffle \
  --dispFreq 500 \
  --sampleFreq 100000 \
  --beamFreq 500 \
  --saveFreq 500 \
  --max_epochs 50 \
  --no_reload_training_progress \
  --n_words_src 60000 \
  --n_words 60000 \
  --reload ${modelDir}/${L1}2${L2}${modelDirSuffix}/model.${L1}2${L2}.npz-160000 \
  --reload_rev ${modelDir}/${L2}2${L1}${modelDirSuffix}/model.${L2}2${L1}.npz-80000 

  # --reload latest_checkpoint \
  # --reload_rev latest_checkpoint \
  #--reinforce \
  #--alpha 0.2 \
  #--valid_datasets $dataPath1/newsdev2016.bpe.${L1} $dataPath1/newsdev2016.bpe.${L2} \
