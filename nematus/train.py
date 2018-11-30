#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Build a neural machine translation model with soft attention
'''
import argparse
import collections
from datetime import datetime
import json
import os
import logging
import subprocess
import sys
import tempfile
import time

import numpy
import tensorflow as tf

from data_iterator import TextIterator
import exception
import inference
import model_loader
from model_updater import ModelUpdater
import rnn_model
import util

import lm
import copy

def create_language_model(config,choice):
    if choice == 'A':
        model_file_name = config.LMA
    elif choice == 'B':
        model_file_name = config.LMB
    else :
        logging.error("How do you get here?create_language_model")
        exit(1)
    assert os.path.exists(model_file_name),'file: '+model_file_name+' not exists!'
    lm_factory = lm.lm_factory(model_file_name)
    return lm_factory

def load_data(config):
    logging.info('Reading data...')
    text_iterator = TextIterator(
                        source=config.source_dataset,
                        target=config.target_dataset,
                        source_dicts=config.source_dicts,
                        target_dict=config.target_dict,
                        batch_size=config.batch_size,
                        maxlen=config.maxlen,
                        source_vocab_sizes=config.source_vocab_sizes,
                        target_vocab_size=config.target_vocab_size,
                        skip_empty=True,
                        shuffle_each_epoch=config.shuffle_each_epoch,
                        sort_by_length=config.sort_by_length,
                        use_factor=(config.factors > 1),
                        maxibatch_size=config.maxibatch_size,
                        token_batch_size=config.token_batch_size,
                        keep_data_in_memory=config.keep_train_set_in_memory)

    if config.validFreq and config.valid_source_dataset and config.valid_target_dataset:
        valid_text_iterator = TextIterator(
                            source=config.valid_source_dataset,
                            target=config.valid_target_dataset,
                            source_dicts=config.source_dicts,
                            target_dict=config.target_dict,
                            batch_size=config.valid_batch_size,
                            maxlen=config.maxlen,
                            source_vocab_sizes=config.source_vocab_sizes,
                            target_vocab_size=config.target_vocab_size,
                            shuffle_each_epoch=False,
                            sort_by_length=True,
                            use_factor=(config.factors > 1),
                            maxibatch_size=config.maxibatch_size,
                            token_batch_size=config.valid_token_batch_size)
    else:
        logging.info('no validation set loaded')
        valid_text_iterator = None
    logging.info('Done')
    return text_iterator, valid_text_iterator


def load_mono_data(config):
    logging.info('Reading data...')
    text_iterator = TextIterator(
                        source=config.source_dataset_mono,
                        target=config.target_dataset_mono,
                        source_dicts=config.source_dicts,
                        target_dict=config.target_dict,
                        batch_size=config.batch_size,
                        maxlen=config.maxlen,
                        source_vocab_sizes=config.source_vocab_sizes,
                        target_vocab_size=config.target_vocab_size,
                        skip_empty=True,
                        shuffle_each_epoch=config.shuffle_each_epoch,
                        sort_by_length=config.sort_by_length,
                        use_factor=(config.factors > 1),
                        maxibatch_size=config.maxibatch_size,
                        token_batch_size=config.token_batch_size,
                        keep_data_in_memory=config.keep_train_set_in_memory)

    if config.validFreq and config.valid_source_dataset and config.valid_target_dataset:
        valid_text_iterator = TextIterator(
                            source=config.valid_source_dataset,
                            target=config.valid_target_dataset,
                            source_dicts=config.source_dicts,
                            target_dict=config.target_dict,
                            batch_size=config.valid_batch_size,
                            maxlen=config.maxlen,
                            source_vocab_sizes=config.source_vocab_sizes,
                            target_vocab_size=config.target_vocab_size,
                            shuffle_each_epoch=False,
                            sort_by_length=True,
                            use_factor=(config.factors > 1),
                            maxibatch_size=config.maxibatch_size,
                            token_batch_size=config.valid_token_batch_size)
    else:
        logging.info('no validation set loaded')
        valid_text_iterator = None
    logging.info('Done')
    return text_iterator, valid_text_iterator


def train(config, sess):
    assert (config.prior_model != None and (tf.train.checkpoint_exists(os.path.abspath(config.prior_model))) or (config.map_decay_c==0.0)), \
    "MAP training requires a prior model file: Use command-line option --prior_model"

    # Construct the graph, with one model replica per GPU

    num_gpus = len(util.get_available_gpus())
    num_replicas = max(1, num_gpus)

    logging.info('Building model...')
    replicas = []
    for i in range(num_replicas):
        device_type = "GPU" if num_gpus > 0 else "CPU"
        device_spec = tf.DeviceSpec(device_type=device_type, device_index=i)
        with tf.device(device_spec):
            with tf.variable_scope(tf.get_variable_scope(), reuse=(i>0)):
                replicas.append(rnn_model.RNNModel(config))

    if config.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate,
                                           beta1=config.adam_beta1,
                                           beta2=config.adam_beta2,
                                           epsilon=config.adam_epsilon)
    else:
        logging.error('No valid optimizer defined: {}'.format(config.optimizer))
        sys.exit(1)

    init = tf.zeros_initializer(dtype=tf.int32)
    global_step = tf.get_variable('time', [], initializer=init, trainable=False)

    if config.summaryFreq:
        summary_dir = (config.summary_dir if config.summary_dir is not None
                       else os.path.abspath(os.path.dirname(config.saveto)))
        writer = tf.summary.FileWriter(summary_dir, sess.graph)
    else:
        writer = None

    updater = ModelUpdater(config, num_gpus, replicas, optimizer, global_step,
                           writer)

    saver, progress = model_loader.init_or_restore_variables(
        config, sess, train=True)

    global_step.load(progress.uidx, sess)

    # Use an InferenceModelSet to abstract over model types for sampling and
    # beam search. Multi-GPU sampling and beam search are not currently
    # supported, so we just use the first replica.
    model_set = inference.InferenceModelSet([replicas[0]], [config])# 这里有问题

    #save model options
    config_as_dict = collections.OrderedDict(sorted(vars(config).items()))
    json.dump(config_as_dict, open('%s.json' % config.saveto, 'wb'), indent=2)

    for model in models:
        text_iterator, valid_text_iterator = load_data(config)
        _, _, num_to_source, num_to_target = util.load_dictionaries(config)
        total_loss = 0.
        n_sents, n_words = 0, 0
    last_time = time.time()
    logging.info("Initial uidx={}".format(progress.uidx))
    for progress.eidx in xrange(progress.eidx, config.max_epochs):
        logging.info('Starting epoch {0}'.format(progress.eidx))
        for source_sents, target_sents in text_iterator:
            if len(source_sents[0][0]) != config.factors:
                logging.error('Mismatch between number of factors in settings ({0}), and number in training corpus ({1})\n'.format(config.factors, len(source_sents[0][0])))
                sys.exit(1)
            x_in, x_mask_in, y_in, y_mask_in = util.prepare_data(
                source_sents, target_sents, config.factors, maxlen=None)
            if x_in is None:
                logging.info('Minibatch with zero sample under length {0}'.format(config.maxlen))
                continue
            write_summary_for_this_batch = config.summaryFreq and ((progress.uidx % config.summaryFreq == 0) or (config.finish_after and progress.uidx % config.finish_after == 0))
            (factors, seqLen, batch_size) = x_in.shape

            loss = updater.update(sess, x_in, x_mask_in, y_in, y_mask_in,
                                  write_summary_for_this_batch)
            total_loss += loss
            n_sents += batch_size
            n_words += int(numpy.sum(y_mask_in))
            progress.uidx += 1

            if config.dispFreq and progress.uidx % config.dispFreq == 0:
                duration = time.time() - last_time
                disp_time = datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')
                logging.info('{0} Epoch: {1} Update: {2} Loss/word: {3} Words/sec: {4} Sents/sec: {5}'.format(disp_time, progress.eidx, progress.uidx, total_loss/n_words, n_words/duration, n_sents/duration))
                last_time = time.time()
                total_loss = 0.
                n_sents = 0
                n_words = 0

            if config.sampleFreq and progress.uidx % config.sampleFreq == 0:
                x_small, x_mask_small, y_small = x_in[:, :, :10], x_mask_in[:, :10], y_in[:, :10]
                samples = model_set.sample(sess, x_small, x_mask_small)
                assert len(samples) == len(x_small.T) == len(y_small.T), (len(samples), x_small.shape, y_small.shape)
                for xx, yy, ss in zip(x_small.T, y_small.T, samples):
                    source = util.factoredseq2words(xx, num_to_source)
                    target = util.seq2words(yy, num_to_target)
                    sample = util.seq2words(ss, num_to_target)
                    logging.info('SOURCE: {}'.format(source))
                    logging.info('TARGET: {}'.format(target))
                    logging.info('SAMPLE: {}'.format(sample))

            if config.beamFreq and progress.uidx % config.beamFreq == 0:
                x_small, x_mask_small, y_small = x_in[:, :, :10], x_mask_in[:, :10], y_in[:,:10]
                samples = model_set.beam_search(sess, x_small, x_mask_small,
                                               config.beam_size)
                # samples is a list with shape batch x beam x len
                assert len(samples) == len(x_small.T) == len(y_small.T), (len(samples), x_small.shape, y_small.shape)
                for xx, yy, ss in zip(x_small.T, y_small.T, samples):
                    source = util.factoredseq2words(xx, num_to_source)
                    target = util.seq2words(yy, num_to_target)
                    logging.info('SOURCE: {}'.format(source))
                    logging.info('TARGET: {}'.format(target))
                    for i, (sample_seq, cost) in enumerate(ss):
                        sample = util.seq2words(sample_seq, num_to_target)
                        msg = 'SAMPLE {}: {} Cost/Len/Avg {}/{}/{}'.format(
                            i, sample, cost, len(sample), cost/len(sample))
                        logging.info(msg)

            if config.validFreq and progress.uidx % config.validFreq == 0:
                costs = validate(config, sess, valid_text_iterator, replicas[0])
                # validation loss is mean of normalized sentence log probs
                valid_loss = sum(costs) / len(costs)
                if (len(progress.history_errs) == 0 or
                    valid_loss < min(progress.history_errs)):
                    progress.history_errs.append(valid_loss)
                    progress.bad_counter = 0
                    saver.save(sess, save_path=config.saveto)
                    progress_path = '{0}.progress.json'.format(config.saveto)
                    progress.save_to_json(progress_path)
                else:
                    progress.history_errs.append(valid_loss)
                    progress.bad_counter += 1
                    if progress.bad_counter > config.patience:
                        logging.info('Early Stop!')
                        progress.estop = True
                        break
                if config.valid_script is not None:
                    score = validate_with_script(sess, replicas[0], config)
                    need_to_save = (score is not None and
                        (len(progress.valid_script_scores) == 0 or
                         score > max(progress.valid_script_scores)))
                    if score is None:
                        score = 0.0  # ensure a valid value is written
                    progress.valid_script_scores.append(score)
                    if need_to_save:
                        save_path = config.saveto + ".best-valid-script"
                        saver.save(sess, save_path=save_path)
                        progress_path = '{}.progress.json'.format(save_path)
                        progress.save_to_json(progress_path)

            if config.saveFreq and progress.uidx % config.saveFreq == 0:
                saver.save(sess, save_path=config.saveto, global_step=progress.uidx)
                progress_path = '{0}-{1}.progress.json'.format(config.saveto, progress.uidx)
                progress.save_to_json(progress_path)

            if config.finish_after and progress.uidx % config.finish_after == 0:
                logging.info("Maximum number of updates reached")
                saver.save(sess, save_path=config.saveto, global_step=progress.uidx)
                progress.estop=True
                progress_path = '{0}-{1}.progress.json'.format(config.saveto, progress.uidx)
                progress.save_to_json(progress_path)
                break
        if progress.estop:
            break


def train_dual(configs):
    assert (configs['A'].prior_model != None and \
        (tf.train.checkpoint_exists(os.path.abspath(configs['A'].prior_model))) or (configs['A'].map_decay_c==0.0)), \
    "MAP training requires a prior model file: Use command-line option --prior_model"

    models = ['A','B']
    def ano(now):
        return 'A' if now == 'B' else 'B'

    # Create the TensorFlow session.
    # gpu_options = tf.GPUOptions(allow_growth=True) # cause fragments
    tf_config = tf.ConfigProto() # gpu_options=gpu_options)
    tf_config.allow_soft_placement = True
    graph = {'A':tf.Graph(),
            'B':tf.Graph()}
    sess = {'A':tf.Session(config=tf_config, graph = graph['A']),
            'B':tf.Session(config=tf_config, graph = graph['B'])}

    # Construct the graph, with one model replica per GPU

    num_gpus = len(util.get_available_gpus())
    num_replicas = max(1, num_gpus)

    logging.info('Building model...')
    replicas =  {'A':[],'B':[]}
    for model in models:
        for i in range(num_replicas):
            device_type = "GPU" if num_gpus > 0 else "CPU"
            device_spec = tf.DeviceSpec(device_type=device_type, device_index=i)
            with graph[model].as_default():
                with tf.device(device_spec):
                    with tf.variable_scope(tf.get_variable_scope(), reuse=(i>0)):
                        replicas[model].append(rnn_model.RNNModel(configs[model]))

    # define tools
    lm,model_set, lm_score_mean,lm_score_std,bt_score_mean,bt_score_std,score_num,\
    writer,summary_dir,optimizer,updater,saver,progress,\
    text_iterator,valid_text_iterator, para_text_iterator, para_valid_text_iterator ,num_to_source, num_to_target,\
    init, global_step,total_loss,\
    n_sents, n_words,source_sents,target_sents,para_source_sents,para_target_sents = \
    dict(),dict(),dict(),dict(),dict(),dict(),dict(),dict(),dict(),dict(),dict(),dict(),dict(),dict(),\
    dict(),dict(),dict(),dict(),dict(),dict(),dict(),dict(),dict(),dict(),dict(),dict(),dict(),dict()

    for model in models:
        with graph[model].as_default():
            lm[model] = model_loader.load_language_model(configs['dual']['lm'][model])
            lm_score_mean[model] = 0
            lm_score_std[model] = 0
            bt_score_mean[model] = 0
            bt_score_std[model] = 0
            score_num[model] = 0

            if configs[model].optimizer == 'adam':
                optimizer[model] = tf.train.AdamOptimizer(learning_rate=configs[model].learning_rate,
                                                   beta1=configs[model].adam_beta1,
                                                   beta2=configs[model].adam_beta2,
                                                   epsilon=configs[model].adam_epsilon)
            else:
                logging.error('No valid optimizer defined: {}'.format(configs[model].optimizer))
                sys.exit(1)

            init[model] = tf.zeros_initializer(dtype=tf.int32)
            global_step[model] = tf.get_variable('time', [], initializer=init[model], trainable=False)

            if configs[model].summaryFreq:
                summary_dir[model] = (os.path.abspath(os.path.dirname(configs[model].saveto)))
                writer[model] = tf.summary.FileWriter(summary_dir[model], sess[model].graph)
            else:
                writer[model] = None

            updater[model] = ModelUpdater(configs[model], num_gpus, replicas[model], optimizer[model], global_step[model],
                                   writer[model])

            saver[model], progress[model] = model_loader.init_or_restore_variables(
                    configs[model], sess[model], train=True)

            global_step[model].load(progress[model].uidx, sess[model])

            # Use an InferenceModelSet to abstract over model types for sampling and
            # beam search. Multi-GPU sampling and beam search are not currently
            # supported, so we just use the first replica.
            model_set[model] = inference.InferenceModelSet([replicas[model][0]], [configs[model]])

            text_iterator[model], valid_text_iterator[model] = load_mono_data(configs[model])
            para_text_iterator[model], para_valid_text_iterator[model] =  load_data(configs[model])  
            _, _, num_to_source[model], num_to_target[model] = util.load_dictionaries(configs[model])

            total_loss[model] = 0.
            n_sents[model], n_words[model] = 0, 0

            #save model options
            config_as_dict = collections.OrderedDict(sorted(vars(configs[model]).items()))
            json.dump(config_as_dict, open('%s.json' % configs[model].saveto, 'wb'), indent=2)

    last_time = time.time()
    logging.info("Initial uidx for modelA ={}".format(progress['A'].uidx))
    logging.info("Initial uidx for modelB ={}".format(progress['B'].uidx))
    # does not update globe epoch in model
    for epoch in range(configs['A'].max_epochs):
        logging.info('Starting epoch {0}'.format(epoch))
        #for source_sents, target_sents in text_iterator:
        for model in models:
            text_iterator[model].reset()
            para_text_iterator[model].reset()
        end_of_iter = False

        # train by using para datasets
        if (configs['dual']['para']):
            logging.info('using para data now...')
            while True:
                for model in models:
                    para_source_sents[model],para_target_sents[model] = [],[]
                    for num in range(configs[model].beam_size):
                        try:
                            ss,tt = para_text_iterator[model].next()
                            for per_ss,per_tt in zip(ss,tt):
                                 para_source_sents[model].append(per_ss)
                                 para_target_sents[model].append(per_tt)
                        except StopIteration:
                            end_of_iter = True
                            break
                if end_of_iter:
                    break
                for model in models:
                    if len(para_source_sents[model][0][0]) != configs[model].factors:
                        logging.error('Mismatch between number of factors in settings ({0}), and number in training corpus ({1})\n'.format(config.factors, len(source_sents[0][0])))
                        sys.exit(1)
                    x_in, x_mask_in, y_in, y_mask_in = util.prepare_data(
                            para_source_sents[model], para_target_sents[model], configs[model].factors, maxlen=None)
                    if x_in is None:
                        logging.info('Minibatch with zero sample under length {0}'.format(configs[model].maxlen))
                        continue
                    write_summary_for_this_batch = configs[model].summaryFreq and ((progress[model].uidx % configs[model].summaryFreq == 0) or (configs[model].finish_after and progress[model].uidx % configs[model].finish_after == 0))
                    (factors, seqLen, batch_size) = x_in.shape

                    with graph[model].as_default():
                        loss = updater[model].update(sess[model],x_in,x_mask_in,y_in,y_mask_in,write_summary_for_this_batch)
                    total_loss[model] += loss
                    n_sents[model] += batch_size
                    n_words[model] += int(numpy.sum(y_mask_in))
                    progress[model].uidx += 1

                    if configs[model].dispFreq and progress[model].uidx % configs[model].dispFreq == 0:
                        duration = time.time() - last_time
                        disp_time = datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')
                        logging.info('{0} Epoch: {1} Update: {2} Loss/word: {3} Words/sec: {4} Sents/sec: {5}'\
                            .format(disp_time, progress[model].eidx, progress[model].uidx, \
                                total_loss[model]/n_words[model], n_words[model]/duration, n_sents[model]/duration))
                        last_time = time.time()
                        total_loss[model] = 0.
                        n_sents[model] = 0
                        n_words[model] = 0

                    if configs[model].sampleFreq and progress[model].uidx % configs[model].sampleFreq == 0:
                        x_small, x_mask_small, y_small = x_in[:, :, :configs[model].beam_size], x_mask_in[:, :configs[model].beam_size], y_in[:, :configs[model].beam_size]
                        with graph[model].as_default():
                            samples = model_set[model].sample(sess[model], x_small, x_mask_small)
                        assert len(samples) == len(x_small.T) == len(y_small.T), \
                        (len(samples), x_small.shape, y_small.shape)
                        for xx, yy, ss in zip(x_small.T, y_small.T, samples):
                            source = util.factoredseq2words(xx, num_to_source[model])
                            target = util.seq2words(yy, num_to_target[model])
                            sample = util.seq2words(ss, num_to_target[model])
                            logging.info('SOURCE: {}'.format(source))
                            logging.info('TARGET: {}'.format(target))
                            logging.info('SAMPLE: {}'.format(sample))


        # train by using mono datasets
        logging.info('para data has been used out at uidx{}, now start using mono data. '.format(progress['A'].uidx))
        end_of_iter = False
        while True:
            for model in models:
                try:
                    source_sents[model],target_sents[model] = text_iterator[model].next()
                except StopIteration:
                    end_of_iter = True
                    break
            if end_of_iter:
                break
            for model in models:
                #logging.info("source now:{}".format(model))
                if len(source_sents[model][0][0]) != configs[model].factors:
                    logging.error('Mismatch between number of factors in settings ({0}), and number in training corpus ({1})\n'.format(config.factors, len(source_sents[0][0])))
                    sys.exit(1)
                x_in, x_mask_in, y_in, y_mask_in = util.prepare_data(
                    source_sents[model], target_sents[model], configs[model].factors, maxlen=None)
                y_in_for_x,y_mask_in_for_x,x_in_for_y,x_mask_in_for_y = util.prepare_data(
                    source_sents[ano(model)], target_sents[ano(model)],configs[model].factors, maxlen=None)
                if x_in is None:
                    logging.info('Minibatch with zero sample under length {0}'.format(configs[model].maxlen))
                    continue
                write_summary_for_this_batch = configs[model].summaryFreq and \
                    ((progress[model].uidx % configs[model].summaryFreq == 0) or \
                        (configs[model].finish_after and progress[model].uidx % configs[model].finish_after == 0))
                (factors, seqLen, batch_size) = x_in.shape


                # logging.info(util.factoredseq2words(source_sents[model][-1],num_to_source[model]))
                # rk_tmp = [1]
                # for t in range(10):
                #     logging.info("-------------------------------------------")

                #     with graph[model].as_default():
                #         smids = model_set[model].beam_search(sess[model], x_in, x_mask_in,
                #                                        configs[model].beam_size) # batch x beam x (len,cost)
                #     smids_flatten = []
                #     multi_source_sents = []
                #     for i,item in enumerate(smids):
                #         for (sent,cost) in item :
                #             smids_flatten.append(sent)
                #             multi_source_sents.append(source_sents[model][i])
                #             # logging.info(sent)
                #             # logging.info(source_sents[model][i])
                #     logging.info(util.seq2words(sent,num_to_target[model]))

                #     multi_x_in_for_x, multi_x_in_mask_for_x, smids_prepared_for_y, smids_prepared_mask_for_y = \
                #         util.prepare_data(multi_source_sents,smids_flatten,configs[model].factors, maxlen=None)
                #     with graph[model].as_default():
                #         loss = updater[model].update_with_rk(sess[model], 
                #                         multi_x_in_for_x, multi_x_in_mask_for_x,
                #                         smids_prepared_for_y, smids_prepared_mask_for_y,
                #                         rk_tmp*(len(smids_flatten)),
                #                         write_summary_for_this_batch)

                # logging.info(model_set[model].get_loss(sess[model],multi_x_in_for_x, multi_x_in_mask_for_x, smids_prepared_for_y, smids_prepared_mask_for_y,rk_tmp*(len(smids_flatten))) )
                # logging.info(model_set[model].get_losses(sess[model],multi_x_in_for_x, multi_x_in_mask_for_x, smids_prepared_for_y, smids_prepared_mask_for_y,rk_tmp*(len(smids_flatten))) )
                # exit()

                # got rk
                rk = []
                lm_scores = []
                bt_scores = []

                smids_flatten = []
                multi_source_sents = []
                with graph[model].as_default():
                    smids = model_set[model].beam_search(sess[model], x_in, x_mask_in,
                                                        configs[model].beam_size) # batch x beam x (len,cost)
                for i,item in enumerate(smids):
                    for (sent,cost) in item :
                        smids_flatten.append(sent)
                        multi_source_sents.append(source_sents[model][i])

                multi_x_in_for_x, multi_x_in_mask_for_x, smids_prepared_for_y, smids_prepared_mask_for_y = \
                    util.prepare_data(multi_source_sents,smids_flatten,configs[model].factors, maxlen=None)
                smids_prepared_for_x, smids_prepared_mask_for_x, multi_x_in_for_y, multi_x_in_mask_for_y = \
                    numpy.array([smids_prepared_for_y]), smids_prepared_mask_for_y, multi_x_in_for_x[0], multi_x_in_mask_for_x
                for i,sample in enumerate(smids_flatten): 
                    # got lm score
                    sent = util.seq2words(sample,num_to_target[model])
                    lm_score = lm[ano(model)].score([sent])[0]
                    lm_scores.append(lm_score)
                    # got bt score 
                    tmp_x, tmp_x_mask, tmp_y, tmp_y_mask = \
                        util.prepare_data([util.wrap_elements(sample)],[util.unwrap_elements(multi_source_sents[i])],\
                            configs[model].factors, maxlen=None)
                    with graph[ano(model)].as_default():
                        bt_scores.append(model_set[ano(model)].get_losses(sess[ano(model)], \
                        tmp_x, tmp_x_mask, tmp_y, tmp_y_mask)[0])

                # should be standardlized, here I update mean and std online
                for lm_sco,bt_sco in zip(lm_scores,bt_scores):
                    lm_score_std[model] = util.update_std(lm_score_std[model],lm_score_mean[model],score_num[model],lm_sco)
                    lm_score_mean[model] = util.update_mean(lm_score_mean[model],score_num[model],lm_sco)
                    bt_score_std[model] = util.update_std(bt_score_std[model],bt_score_mean[model],score_num[model],bt_sco)
                    bt_score_mean[model] = util.update_mean(bt_score_mean[model],score_num[model],bt_sco)
                    score_num[model] = util.update_num(score_num[model], 1)
                lm_scores = util.standardlize(lm_scores,lm_score_mean[model],lm_score_std[model])
                bt_scores = util.standardlize(bt_scores,bt_score_mean[model],bt_score_std[model])
                for lm_score,bt_score in zip(lm_scores,bt_scores):
                    rk.append( (lm_score*configs['dual']['alpha'] + bt_score*(1-configs['dual']['alpha'])) )

                # dual learning
                if configs['dual']['reinforce']:
                    with graph[model].as_default():
                        loss = updater[model].update_with_rk(sess[model], 
                                        multi_x_in_for_x, multi_x_in_mask_for_x,
                                        smids_prepared_for_y, smids_prepared_mask_for_y,
                                        rk,
                                        write_summary_for_this_batch)
                            
                    total_loss[model] += loss
                    n_sents[model] += batch_size
                    n_words[model] += int(numpy.sum(smids_prepared_mask_for_y))

                with graph[ano(model)].as_default():
                    loss_ano = updater[ano(model)].update_with_rk(sess[ano(model)],
                                    smids_prepared_for_x, smids_prepared_mask_for_x, 
                                    multi_x_in_for_y, multi_x_in_mask_for_y,
                                    [1-configs['dual']['alpha']]*len(rk),
                                    write_summary_for_this_batch)

                progress[model].uidx += 1

                if configs[model].dispFreq and progress[model].uidx % configs[model].dispFreq == 0:
                    duration = time.time() - last_time
                    disp_time = datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')
                    logging.info('{0} Epoch: {1} Update: {2} Loss/word: {3} Words/sec: {4} Sents/sec: {5}'\
                        .format(disp_time, progress[model].eidx, progress[model].uidx, \
                            total_loss[model]/n_words[model], n_words[model]/duration, n_sents[model]/duration))
                    last_time = time.time()
                    total_loss[model] = 0.
                    n_sents[model] = 0
                    n_words[model] = 0

                if configs[model].sampleFreq and progress[model].uidx % configs[model].sampleFreq == 0:
                    logging.info("lm_scores: {}  mean: {}  std: {}\n bt_scores: {}  mean: {}  std: {}\n rk: {}  score_num: {}"\
                        .format('[lm_scores is pass. see nmt.py 556]', lm_score_mean[model],lm_score_std[model], bt_scores, bt_score_mean[model], bt_score_std[model], rk, score_num[model]))

                    x_small, x_mask_small, y_small = multi_x_in_for_x[:, :, :configs[model].beam_size], multi_x_in_mask_for_x[:, :configs[model].beam_size], smids_prepared_for_y[:, :configs[model].beam_size]
                    with graph[model].as_default():
                        samples = model_set[model].sample(sess[model], x_small, x_mask_small)
                    assert len(samples) == len(x_small.T) == len(y_small.T), \
                    (len(samples), x_small.shape, y_small.shape)
                    for xx, yy, ss in zip(x_small.T, y_small.T, samples):
                        source = util.factoredseq2words(xx, num_to_source[model])
                        target = util.seq2words(yy, num_to_target[model])
                        sample = util.seq2words(ss, num_to_target[model])
                        logging.info('SOURCE: {}'.format(source))
                        logging.info('BEFORE: {}'.format(target))
                        logging.info('AFTER: {}'.format(sample))
                        logging.info('-----------------------------------------------')

                if configs[model].beamFreq and progress[model].uidx % configs[model].beamFreq == 0:
                    x_small, x_mask_small, y_small = multi_x_in_for_x[:, :, :configs[model].beam_size], multi_x_in_mask_for_x[:, :configs[model].beam_size], smids_prepared_for_y[:,:configs[model].beam_size]
                    with graph[model].as_default():
                        samples = model_set[model].beam_search(sess[model], x_small, x_mask_small,
                                                   configs[model].beam_size)
                    # samples is a list with shape batch x beam x len
                    assert len(samples) == len(x_small.T) == len(y_small.T), (len(samples), x_small.shape, y_small.shape)
                    for xx, yy, ss in zip(x_small.T, y_small.T, samples):
                        source = util.factoredseq2words(xx, num_to_source[model])
                        target = util.seq2words(yy, num_to_target[model])
                        logging.info('SOURCE: {}'.format(source))
                        logging.info('BEFORE: {}'.format(target))
                        for i, (sample_seq, cost) in enumerate(ss):
                            sample = util.seq2words(sample_seq, num_to_target[model])
                            msg = 'AFTER {}: {} Cost/Len/Avg {}/{}/{}'.format(
                                i, sample, cost, len(sample), cost/len(sample))
                            logging.info(msg)
                        logging.info('-----------------------------------------------')

                if configs[model].validFreq and progress[model].uidx % configs[model].validFreq == 0:
                    with graph[model].as_default():
                        costs = validate(configs[model], sess[model], valid_text_iterator[model], replicas[model][0])
                    # validation loss is mean of normalized sentence log probs
                    valid_loss = sum(costs) / len(costs)
                    if (len(progress.history_errs) == 0 or
                        valid_loss < min(progress.history_errs)):
                        progress[model].history_errs.append(valid_loss)
                        progress[model].bad_counter = 0
                        with graph[model].as_default():
                            saver.save(sess[model], save_path=configs[model].saveto)
                        progress_path = '{0}.progress.json'.format(configs[model].saveto)
                        progress.save_to_json(progress_path)
                    else:
                        progress[model].history_errs.append(valid_loss)
                        progress[model].bad_counter += 1
                        if progress[model].bad_counter > configs[model].patience:
                            logging.info('Early Stop!')
                            progress[model].estop = True
                            break
                    if configs[model].valid_script is not None:
                        with graph[model].as_default():
                            score = validate_with_script(sess[model], replicas[model][0], configs[model])
                        need_to_save = (score is not None and
                            (len(progress.valid_script_scores) == 0 or
                             score > max(progress.valid_script_scores)))
                        if score is None:
                            score = 0.0  # ensure a valid value is written
                        progress[model].valid_script_scores.append(score)
                        if need_to_save:
                            save_path = configs[model].saveto + ".best-valid-script"
                            with graph[model].as_default():
                                saver[model].save(sess[model], save_path=save_path)
                            progress_path = '{}.progress.json'.format(save_path)
                            progress[model].save_to_json(progress_path)

                if configs[model].saveFreq and progress[model].uidx % configs[model].saveFreq == 0:
                    with graph[model].as_default():
                        saver[model].save(sess[model], save_path=configs[model].saveto, global_step=progress[model].uidx)
                    progress_path = '{0}-{1}.progress.json'.format(configs[model].saveto, progress[model].uidx)
                    progress[model].save_to_json(progress_path)

                if configs[model].finish_after and progress[model].uidx % configs[model].finish_after == 0:
                    logging.info("Maximum number of updates reached")
                    with graph[model].as_default():
                        saver[model].save(sess[model], save_path=configs[model].saveto, global_step=progress[model].uidx)
                    progress[model].estop=True
                    progress_path = '{0}-{1}.progress.json'.format(configs[model].saveto, progress[model].uidx)
                    progress[model].save_to_json(progress_path)
                    break

        if progress[model].estop:
            break


def validate_with_script(sess, model, config):
    if config.valid_script == None:
        return None
    logging.info('Starting external validation.')
    out = tempfile.NamedTemporaryFile()
    inference.translate_file(input_file=open(config.valid_source_dataset),
                             output_file=out,
                             session=sess,
                             models=[model],
                             configs=[config],
                             beam_size=config.beam_size,
                             minibatch_size=config.valid_batch_size,
                             normalization_alpha=1.0)
    out.flush()
    args = [config.valid_script, out.name]
    proc = subprocess.Popen(args, stdin=None, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    stdout, stderr = proc.communicate()
    if len(stderr) > 0:
        logging.info("Validation script wrote the following to standard "
                     "error:\n" + stderr)
    if proc.returncode != 0:
        logging.warning("Validation script failed (returned exit status of "
                        "{}).".format(proc.returncode))
        return None
    try:
        score = float(stdout.split()[0])
    except:
        logging.warning("Validation script output does not look like a score: "
                        "{}".format(stdout))
        return None
    logging.info("Validation script score: {}".format(score))
    return score


# TODO Support for multiple GPUs
def calc_loss_per_sentence(config, sess, text_iterator, model,
                           normalization_alpha=0):
    losses = []
    for x_v, y_v in text_iterator:
        if len(x_v[0][0]) != config.factors:
            logging.error('Mismatch between number of factors in settings ({0}), and number in validation corpus ({1})\n'.format(config.factors, len(x_v[0][0])))
            sys.exit(1)
        x, x_mask, y, y_mask = util.prepare_data(x_v, y_v, config.factors,
                                                 maxlen=None)
        feeds = {model.inputs.x: x,
                 model.inputs.x_mask: x_mask,
                 model.inputs.y: y,
                 model.inputs.y_mask: y_mask,
                 model.inputs.training: False}
        loss_per_sentence = sess.run(model.loss_per_sentence, feed_dict=feeds)

        # normalize scores according to output length
        if normalization_alpha:
            adjusted_lengths = numpy.array([numpy.count_nonzero(s) ** normalization_alpha for s in y_mask.T])
            loss_per_sentence /= adjusted_lengths

        losses += list(loss_per_sentence)
        logging.info( "Seen {0}".format(len(losses)))
    return losses


def validate(config, sess, text_iterator, model, normalization_alpha=0):
    losses = calc_loss_per_sentence(config, sess, text_iterator, model,
                                    normalization_alpha)
    num_sents = len(losses)
    total_loss = sum(losses)
    logging.info('Validation loss (AVG/SUM/N_SENT): {0} {1} {2}'.format(
        total_loss/num_sents, total_loss, num_sents))
    return losses


def parse_args():
    parser = argparse.ArgumentParser()

    data = parser.add_argument_group('data sets; model loading and saving')

    data.add_argument('--source_dataset', type=str, metavar='PATH', 
                         help="parallel training corpus (source)")
    data.add_argument('--target_dataset', type=str, metavar='PATH', 
                         help="parallel training corpus (target)")
    # parallel training corpus (source and target). Hidden option for backward compatibility
    data.add_argument('--datasets', type=str, metavar='PATH', nargs=2,
                         help=argparse.SUPPRESS)
    data.add_argument('--dictionaries', type=str, required=True, metavar='PATH', nargs="+",
                         help="network vocabularies (one per source factor, plus target vocabulary)")
    data.add_argument('--saveFreq', type=int, default=30000, metavar='INT',
                         help="save frequency (default: %(default)s)")
    data.add_argument('--model', '--saveto', type=str, default='model', metavar='PATH', dest='saveto',
                         help="model file name (default: %(default)s)")
    data.add_argument('--reload', type=str, default=None, metavar='PATH',
                         help="load existing model from this path. Set to \"latest_checkpoint\" to reload the latest checkpoint in the same directory of --model")
    data.add_argument('--no_reload_training_progress', action='store_false',  dest='reload_training_progress',
                         help="don't reload training progress (only used if --reload is enabled)")
    data.add_argument('--summary_dir', type=str, required=False, metavar='PATH', 
                         help="directory for saving summaries (default: same directory as the --model file)")
    data.add_argument('--summaryFreq', type=int, default=0, metavar='INT',
                         help="Save summaries after INT updates, if 0 do not save summaries (default: %(default)s)")

    network = parser.add_argument_group('network parameters')
    network.add_argument('--embedding_size', '--dim_word', type=int, default=512, metavar='INT',
                         help="embedding layer size (default: %(default)s)")
    network.add_argument('--state_size', '--dim', type=int, default=1000, metavar='INT',
                         help="hidden state size (default: %(default)s)")

    network.add_argument('--source_vocab_sizes', '--n_words_src', type=int, default=None, nargs='+', metavar='INT',
                         help="source vocabulary sizes (one per input factor) (default: %(default)s)")

    network.add_argument('--target_vocab_size', '--n_words', type=int, default=-1, metavar='INT',
                         help="target vocabulary size (default: %(default)s)")
    network.add_argument('--factors', type=int, default=1, metavar='INT',
                         help="number of input factors (default: %(default)s)")

    network.add_argument('--dim_per_factor', type=int, default=None, nargs='+', metavar='INT',
                         help="list of word vector dimensionalities (one per factor): '--dim_per_factor 250 200 50' for total dimensionality of 500 (default: %(default)s)")
    network.add_argument('--enc_depth', type=int, default=1, metavar='INT',
                         help="number of encoder layers (default: %(default)s)")
    network.add_argument('--enc_recurrence_transition_depth', type=int, default=1, metavar='INT',
                         help="number of GRU transition operations applied in the encoder. Minimum is 1. (Only applies to gru). (default: %(default)s)")
    network.add_argument('--dec_depth', type=int, default=1, metavar='INT',
                         help="number of decoder layers (default: %(default)s)")
    network.add_argument('--dec_base_recurrence_transition_depth', type=int, default=2, metavar='INT',
                         help="number of GRU transition operations applied in the first layer of the decoder. Minimum is 2.  (Only applies to gru_cond). (default: %(default)s)")
    network.add_argument('--dec_high_recurrence_transition_depth', type=int, default=1, metavar='INT',
                         help="number of GRU transition operations applied in the higher layers of the decoder. Minimum is 1. (Only applies to gru). (default: %(default)s)")
    network.add_argument('--dec_deep_context', action='store_true',
                         help="pass context vector (from first layer) to deep decoder layers")
    network.add_argument('--use_dropout', action="store_true",
                         help="use dropout layer (default: %(default)s)")
    network.add_argument('--dropout_embedding', type=float, default=0.2, metavar="FLOAT",
                         help="dropout for input embeddings (0: no dropout) (default: %(default)s)")
    network.add_argument('--dropout_hidden', type=float, default=0.2, metavar="FLOAT",
                         help="dropout for hidden layer (0: no dropout) (default: %(default)s)")
    network.add_argument('--dropout_source', type=float, default=0.0, metavar="FLOAT",
                         help="dropout source words (0: no dropout) (default: %(default)s)")
    network.add_argument('--dropout_target', type=float, default=0.0, metavar="FLOAT",
                         help="dropout target words (0: no dropout) (default: %(default)s)")
    network.add_argument('--use_layer_norm', '--layer_normalisation', action="store_true", dest="use_layer_norm",
                         help="Set to use layer normalization in encoder and decoder")
    network.add_argument('--tie_encoder_decoder_embeddings', action="store_true", dest="tie_encoder_decoder_embeddings",
                         help="tie the input embeddings of the encoder and the decoder (first factor only). Source and target vocabulary size must be the same")
    network.add_argument('--tie_decoder_embeddings', action="store_true", dest="tie_decoder_embeddings",
                         help="tie the input embeddings of the decoder with the softmax output embeddings")
    network.add_argument('--output_hidden_activation', type=str, default='tanh',
                         choices=['tanh', 'relu', 'prelu', 'linear'],
                         help='activation function in hidden layer of the output network (default: %(default)s)')
    network.add_argument('--softmax_mixture_size', type=int, default=1, metavar="INT",
                         help="number of softmax components to use (default: %(default)s)")

    training = parser.add_argument_group('training parameters')
    training.add_argument('--maxlen', type=int, default=100, metavar='INT',
                         help="maximum sequence length for training and validation (default: %(default)s)")
    training.add_argument('--batch_size', type=int, default=80, metavar='INT',
                         help="minibatch size (default: %(default)s)")
    training.add_argument('--token_batch_size', type=int, default=0, metavar='INT',
                          help="minibatch size (expressed in number of source or target tokens). Sentence-level minibatch size will be dynamic. If this is enabled, batch_size only affects sorting by length. (default: %(default)s)")
    training.add_argument('--max_epochs', type=int, default=5000, metavar='INT',
                         help="maximum number of epochs (default: %(default)s)")
    training.add_argument('--finish_after', type=int, default=100000000, metavar='INT',
                         help="maximum number of updates (minibatches) (default: %(default)s)")
    training.add_argument('--decay_c', type=float, default=0.0, metavar='FLOAT',
                         help="L2 regularization penalty (default: %(default)s)")
    training.add_argument('--map_decay_c', type=float, default=0.0, metavar='FLOAT',
                         help="MAP-L2 regularization penalty towards original weights (default: %(default)s)")
    training.add_argument('--prior_model', type=str, metavar='PATH',
                         help="Prior model for MAP-L2 regularization. Unless using \"--reload\", this will also be used for initialization.")
    training.add_argument('--clip_c', type=float, default=1.0, metavar='FLOAT',
                         help="gradient clipping threshold (default: %(default)s)")
    training.add_argument('--label_smoothing', type=float, default=0.0, metavar='FLOAT',
                         help="label smoothing (default: %(default)s)")
    training.add_argument('--no_shuffle', action="store_false", dest="shuffle_each_epoch",
                         help="disable shuffling of training data (for each epoch)")
    training.add_argument('--keep_train_set_in_memory', action="store_true", 
                         help="Keep training dataset lines stores in RAM during training")
    training.add_argument('--no_sort_by_length', action="store_false", dest="sort_by_length",
                         help='do not sort sentences in maxibatch by length')
    training.add_argument('--maxibatch_size', type=int, default=20, metavar='INT',
                         help='size of maxibatch (number of minibatches that are sorted by length) (default: %(default)s)')
    training.add_argument(
        '--optimizer', type=str, default="adam", choices=['adam'],
        help="optimizer (default: %(default)s)")
    training.add_argument(
        '--learning_rate', '--lrate', type=float, default=0.0001,
        metavar='FLOAT',
        help="learning rate (default: %(default)s)")
    training.add_argument(
        '--adam_beta1', type=float, default=0.9, metavar='FLOAT',
        help='exponential decay rate for the first moment estimates ' \
             '(default: %(default)s)')
    training.add_argument(
        '--adam_beta2', type=float, default=0.999, metavar='FLOAT',
        help='exponential decay rate for the second moment estimates ' \
             '(default: %(default)s)')
    training.add_argument(
        '--adam_epsilon', type=float, default=1e-08, metavar='FLOAT', \
        help='constant for numerical stability (default: %(default)s)')

    validation = parser.add_argument_group('validation parameters')
    validation.add_argument('--valid_source_dataset', type=str, default=None, metavar='PATH', 
                         help="source validation corpus (default: %(default)s)")
    validation.add_argument('--valid_target_dataset', type=str, default=None, metavar='PATH',
                         help="target validation corpus (default: %(default)s)")
    # parallel validation corpus (source and target). Hidden option for backward compatibility
    validation.add_argument('--valid_datasets', type=str, default=None, metavar='PATH', nargs=2,
                         help=argparse.SUPPRESS)
    validation.add_argument('--valid_batch_size', type=int, default=80, metavar='INT',
                         help="validation minibatch size (default: %(default)s)")
    training.add_argument('--valid_token_batch_size', type=int, default=0, metavar='INT',
                          help="validation minibatch size (expressed in number of source or target tokens). Sentence-level minibatch size will be dynamic. If this is enabled, valid_batch_size only affects sorting by length. (default: %(default)s)")
    validation.add_argument('--validFreq', type=int, default=10000, metavar='INT',
                         help="validation frequency (default: %(default)s)")
    validation.add_argument('--valid_script', type=str, default=None, metavar='PATH',
                         help="path to script for external validation (default: %(default)s). The script will be passed an argument specifying the path of a file that contains translations of the source validation corpus. It must write a single score to standard output.")
    validation.add_argument('--patience', type=int, default=10, metavar='INT',
                         help="early stopping patience (default: %(default)s)")

    display = parser.add_argument_group('display parameters')
    display.add_argument('--dispFreq', type=int, default=1000, metavar='INT',
                         help="display loss after INT updates (default: %(default)s)")
    display.add_argument('--sampleFreq', type=int, default=10000, metavar='INT',
                         help="display some samples after INT updates (default: %(default)s)")
    display.add_argument('--beamFreq', type=int, default=10000, metavar='INT',
                         help="display some beam_search samples after INT updates (default: %(default)s)")
    display.add_argument('--beam_size', type=int, default=12, metavar='INT',
                         help="size of the beam (default: %(default)s)")

    translate = parser.add_argument_group('translate parameters')
    translate.add_argument('--no_normalize', action='store_false', dest='normalize',
                            help="Cost of sentences will not be normalized by length")
    translate.add_argument('--n_best', action='store_true', dest='n_best',
                            help="Print full beam")
    translate.add_argument('--translation_maxlen', type=int, default=200, metavar='INT',
                         help="Maximum length of translation output sentence (default: %(default)s)")

    dual = parser.add_argument_group('parameters for dual learning. ')
    dual.add_argument('--dual', action='store_true',  dest='dual',
                         help="active dual learning")
    dual.add_argument('--para', action='store_true',  dest='para',
                         help="active parallel dataset using in dual learning")
    dual.add_argument('--reinforce', action='store_true',  dest='reinforce',
                         help="active reinforcement learning")
    dual.add_argument('--model_rev', '--saveto_rev', type=str, default='model_rev', metavar='PATH', dest='saveto_rev',
                         help="reverse model file name (default: %(default)s)")
    dual.add_argument('--source_lm', type=str, metavar='PATH', 
                         help="language model (source)")
    dual.add_argument('--target_lm', type=str, metavar='PATH', 
                         help="language model (target)")
    dual.add_argument('--lms',type=str, metavar='PATH', nargs="+",
                         help="language models (one for source, and one for target." )
    dual.add_argument('--source_dataset_mono', type=str, metavar='PATH', 
                         help="parallel training corpus (source)")
    dual.add_argument('--target_dataset_mono', type=str, metavar='PATH', 
                         help="parallel training corpus (target)")
    dual.add_argument('--datasets_mono', type=str, metavar='PATH', nargs=2,
                         help=argparse.SUPPRESS)
    dual.add_argument('--alpha', type=float, default=0.5, metavar='FLOAT',
        help='weight for lm score. ')

    config = parser.parse_args()

    # allow "--datasets" for backward compatibility
    if config.datasets:
        if config.source_dataset or config.target_dataset:
            logging.error('argument clash: --datasets is mutually exclusive with --source_dataset and --target_dataset')
            sys.exit(1)
        else:
            config.source_dataset = config.datasets[0]
            config.target_dataset = config.datasets[1]
    elif not config.source_dataset:
        logging.error('--source_dataset is required')
        sys.exit(1)
    elif not config.target_dataset:
        logging.error('--target_dataset is required')
        sys.exit(1)

    # allow "--valid_datasets" for backward compatibility
    if config.valid_datasets:
        if config.valid_source_dataset or config.valid_target_dataset:
            logging.error('argument clash: --valid_datasets is mutually exclusive with --valid_source_dataset and --valid_target_dataset')
            sys.exit(1)
        else:
            config.valid_source_dataset = config.valid_datasets[0]
            config.valid_target_dataset = config.valid_datasets[1]

    # check factor-related options are consistent

    if config.dim_per_factor == None:
        if config.factors == 1:
            config.dim_per_factor = [config.embedding_size]
        else:
            logging.error('if using factored input, you must specify \'dim_per_factor\'\n')
            sys.exit(1)

    if len(config.dim_per_factor) != config.factors:
        logging.error('mismatch between \'--factors\' ({0}) and \'--dim_per_factor\' ({1} entries)\n'.format(config.factors, len(config.dim_per_factor)))
        sys.exit(1)

    if sum(config.dim_per_factor) != config.embedding_size:
        logging.error('mismatch between \'--embedding_size\' ({0}) and \'--dim_per_factor\' (sums to {1})\n'.format(config.embedding_size, sum(config.dim_per_factor)))
        sys.exit(1)

    if len(config.dictionaries) != config.factors + 1:
        logging.error('\'--dictionaries\' must specify one dictionary per source factor and one target dictionary\n')
        sys.exit(1)

    # determine target_embedding_size
    if config.tie_encoder_decoder_embeddings:
        config.target_embedding_size = config.dim_per_factor[0]
    else:
        config.target_embedding_size = config.embedding_size

    # set vocabulary sizes
    vocab_sizes = []
    if config.source_vocab_sizes == None:
        vocab_sizes = [-1] * config.factors
    elif len(config.source_vocab_sizes) == config.factors:
        vocab_sizes = config.source_vocab_sizes
    elif len(config.source_vocab_sizes) < config.factors:
        num_missing = config.factors - len(config.source_vocab_sizes)
        vocab_sizes += config.source_vocab_sizes + [-1] * num_missing
    else:
        logging.error('too many values supplied to \'--source_vocab_sizes\' option (expected one per factor = {0})'.format(config.factors))
        sys.exit(1)
    if config.target_vocab_size == -1:
        vocab_sizes.append(-1)
    else:
        vocab_sizes.append(config.target_vocab_size)

    # for unspecified vocabulary sizes, determine sizes from vocabulary dictionaries
    for i, vocab_size in enumerate(vocab_sizes):
        if vocab_size >= 0:
            continue
        try:
            d = util.load_dict(config.dictionaries[i])
        except:
            logging.error('failed to determine vocabulary size from file: {0}'.format(config.dictionaries[i]))
        vocab_sizes[i] = max(d.values()) + 1

    config.source_dicts = config.dictionaries[:-1]
    config.source_vocab_sizes = vocab_sizes[:-1]
    config.target_dict = config.dictionaries[-1]
    config.target_vocab_size = vocab_sizes[-1]

    # set the model version
    config.model_version = 0.2
    config.theano_compat = False

    configs = {}
    # check parameters for dual learning
    if config.dual:
        assert config.factors == 1, 'In dual learning, factor must be 1.'
        assert config.reload == 'latest_checkpoint'
        config_rev = copy.deepcopy(config)
        configs['dual'] = {'lm':{},'dataset':{}}

        if config.datasets_mono:
            if config.source_dataset_mono or config.target_dataset_mono:
                logging.error('argument clash: --datasets_mono is mutually exclusive with --source_dataset_mono and --target_dataset_mono')
                sys.exit(1)
            else:
                config.source_dataset_mono = config.datasets_mono[0]
                config.target_dataset_mono = config.datasets_mono[1]
                configs['dual']['dataset']['A'] = config.datasets_mono[0]
                configs['dual']['dataset']['B'] = config.datasets_mono[1]
        elif not config.source_dataset_mono:
            logging.error('--source_dataset_mono is required')
            sys.exit(1)
        elif not config.target_dataset_mono:
            logging.error('--target_dataset_mono is required')
            sys.exit(1)
        else:
            configs['dual']['dataset']['A'] = config.source_dataset_mono
            configs['dual']['dataset']['B'] = config.target_dataset_mono

        if config.lms:
            if config.source_lm or config.target_lm:
                logging.error('argument clash: --lms is mutually exclusive with --source_lm and --target_lm')
                sys.exit(1)
            else:
                configs['dual']['lm']['A'] = config.lms[0]
                configs['dual']['lm']['B'] = config.lms[1]
        elif not config.source_lm:
            logging.error('--source_lm is required')
            sys.exit(1)
        elif not config.target_lm:
            logging.error('--target_lm is required')
            sys.exit(1)
        else:
            configs['dual']['lm']['A'] = config.source_lm
            configs['dual']['lm']['B'] = config.target_lm

        if config.alpha>1. or config.alpha<0.:
            logging.error('--alpha should be in range(0,1)')
            sys.exit(1)

        if not config.saveto_rev:
            logging.error('you must specify a reverse model in dual learning')
            sys.exit(1)
        else:
            config_rev.source_dataset = config.target_dataset
            config_rev.target_dataset = config.source_dataset
            config_rev.saveto = config.saveto_rev
            config_rev.saveto_rev = config.saveto
            config_rev.valid_source_dataset = config.valid_target_dataset
            config_rev.valid_target_dataset = config.valid_source_dataset
            config_rev.source_dicts = [config.target_dict]
            config_rev.source_vocab_sizes = [config.target_vocab_size]
            config_rev.target_dict = config.source_dicts[0]
            config_rev.target_vocab_size = config.source_vocab_sizes[0]
            config_rev.source_dataset_mono = config.target_dataset_mono
            config_rev.target_dataset_mono = config.source_dataset_mono

        configs['dual']['alpha'] = config.alpha
        configs['dual']['para'] = False
        configs['dual']['reinforce'] = False
        if config.para:
            configs['dual']['para'] = True
        if config.reinforce:
            configs['dual']['reinforce'] = True


    configs['A'] = config
    configs['B'] = config_rev

    return configs


if __name__ == "__main__":
    # Start logging.
    level = logging.INFO
    logging.basicConfig(level=level, format='%(levelname)s: %(message)s')

    # Parse command-line arguments.
    configs = parse_args()
    # logging.info("{}".format(configs['A']))
    # logging.info("{}".format(configs['B']))

    # Create the TensorFlow session.
    tf_config = tf.ConfigProto()
    tf_config.allow_soft_placement = True

    # Train.
    if configs['A'].dual:
        train_dual(configs)
    else:
        with tf.Session(config=tf_config) as sess:
            train(configs['A'], sess)
