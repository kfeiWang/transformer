# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
June 2017 by kyubyong park. 
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''

from __future__ import print_function
import codecs
import os

import tensorflow as tf
import numpy as np

from hyperparams import Hyperparams as hp
from data_load import load_test_data, load_de_vocab, load_en_vocab
from train import Graph
from nltk.translate.bleu_score import corpus_bleu

def eval(): 
    # Load graph
    g = Graph(is_training=False)
    print("Graph loaded")
    
    # Load data
    X, Sources, Targets = load_test_data()
    de2idx, idx2de = load_de_vocab()
    en2idx, idx2en = load_en_vocab()
#     X, Sources, Targets = X[:33], Sources[:33], Targets[:33]
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    # Start session         
    with g.graph.as_default():    
        sv = tf.train.Supervisor()
        with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)) as sess:
            ## Restore parameters
            sv.saver.restore(sess, tf.train.latest_checkpoint(hp.logdir))
            print("Restored!")
              
            ## Get model name
            mname = open(hp.logdir + '/checkpoint', 'r').read().split('"')[1] # model name
            ## Inference
            totalTransNum = 0
            if not os.path.exists('results'): os.mkdir('results')
            with codecs.open('results/'+mname+'.trans', 'w', 'utf8') as tfout:
                with codecs.open("results/" + mname, "w", "utf-8") as fout:
                    list_of_refs, hypotheses = [], []
                    for i in range((len(X) // hp.batch_size) + 1):
                        ### Get mini-batches
                        batchEnd = (i+1)*hp.batch_size
                        readlBatchSize = hp.batch_size
                        if batchEnd > len(X):
                            readlBatchSize = hp.batch_size - (batchEnd - len(X))
                            batchEnd = len(X)

                        x = X[i*hp.batch_size: batchEnd]
                        sources = Sources[i*hp.batch_size: batchEnd]
                        targets = Targets[i*hp.batch_size: batchEnd]
                        totalTransNum += len(sources)
                        ### Autoregressive inference
                        preds = np.zeros((readlBatchSize, hp.maxlen), np.int32)
                        for j in range(hp.maxlen):
                            _preds = sess.run(g.preds, {g.x: x, g.y: preds})
                            preds[:, j] = _preds[:, j]

                        ### Write to file
                        for source, target, pred in zip(sources, targets, preds): # sentence-wise
                            got = " ".join(idx2en[idx] for idx in pred).split("</S>")[0].strip()
                            fout.write("- source: " + source +"\n")
                            fout.write("- expected: " + target + "\n")
                            fout.write("- got: " + got + "\n\n")
                            tfout.write(got)
                            tfout.write('\n')

                            # bleu score
                            ref = target.split()
                            hypothesis = got.split()
                            if len(ref) > 3 and len(hypothesis) > 3:
                                list_of_refs.append([ref])
                                hypotheses.append(hypothesis)

                    ## Calculate bleu score
                    score = corpus_bleu(list_of_refs, hypotheses)
                    fout.write("Bleu Score = " + str(100*score))
                    fout.write('\n')

                    print('totalTransNum', totalTransNum, 'Bleu', str(100*score))
                                          
if __name__ == '__main__':
    eval()
    print("Done")
    
    