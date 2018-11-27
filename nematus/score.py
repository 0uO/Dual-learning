#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Given a parallel corpus of sentence pairs: with one-to-one of target and source sentences,
produce the score.
"""

import sys
import argparse
import tempfile
import logging

import numpy

from data_iterator import TextIterator
from util import load_config
from compat import fill_options
import model_loader
import train
import rnn_model
from settings import ScorerSettings

import tensorflow as tf

def score_model(source_file, target_file, scorer_settings, options):
    scores = []
    for option in options:
        g = tf.Graph()
        with g.as_default():
            tf_config = tf.ConfigProto()
            tf_config.allow_soft_placement = True
            with tf.Session(config=tf_config) as sess:
                logging.info('Building model...')
                model = rnn_model.RNNModel(option)
                saver = model_loader.init_or_restore_variables(option, sess)

                text_iterator = TextIterator(
                    source=source_file.name,
                    target=target_file.name,
                    source_dicts=option.source_dicts,
                    target_dict=option.target_dict,
                    batch_size=scorer_settings.minibatch_size,
                    maxlen=float('inf'),
                    source_vocab_sizes=option.source_vocab_sizes,
                    target_vocab_size=option.target_vocab_size,
                    use_factor=(option.factors > 1),
                    sort_by_length=False)

                losses = train.calc_loss_per_sentence(
                    option,
                    sess,
                    text_iterator,
                    model,
                    normalization_alpha=scorer_settings.normalization_alpha)

                scores.append(losses)
    return scores


def write_scores(source_file, target_file, scores, output_file, scorer_settings):

    source_file.seek(0)
    target_file.seek(0)
    source_lines = source_file.readlines()
    target_lines = target_file.readlines()

    for i, line in enumerate(target_lines):
        score_str = ' '.join(map(str,[s[i] for s in scores]))
        if scorer_settings.verbose:
            output_file.write('{0} '.format(line.strip()))
        output_file.write('{0}\n'.format(score_str))


def main(source_file, target_file, output_file, scorer_settings):
    # load model model_options
    options = []
    for model in scorer_settings.models:
        options.append(load_config(model))
        fill_options(options[-1])
        options[-1]['reload'] = model
        options[-1] = argparse.Namespace(**options[-1])

    scores = score_model(source_file, target_file, scorer_settings, options)
    write_scores(source_file, target_file, scores, output_file, scorer_settings)

if __name__ == "__main__":
    scorer_settings = ScorerSettings(from_console_arguments=True)
    source_file = scorer_settings.source
    target_file = scorer_settings.target
    output_file = scorer_settings.output
    level = logging.DEBUG if scorer_settings.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(levelname)s: %(message)s')
    main(source_file, target_file, output_file, scorer_settings)
