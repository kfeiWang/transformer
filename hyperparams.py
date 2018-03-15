# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
June 2017 by kyubyong park. 
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''
class Hyperparams:
    '''Hyperparameters'''
    # data
    source_train = 'corpora/train.clean.mo'
    target_train = 'corpora/train.clean.ch'
    source_test = 'corpora/test.clean.mo'
    target_test = 'corpora/test.clean.ch'
    
    # training
    batch_size = 32 # alias = N
    lr = 0.0001 # learning rate. In paper, learning rate is adjusted to the global step.
    logdir = 'logdir' # log directory
    
    # model
    maxlen = 50 # Maximum number of words in a sentence. alias = T.
                # Feel free to increase this if you are ambitious.
    min_cnt = 0 # words whose occurred less than min_cnt are encoded as <UNK>.
    eval_step = 1000
    hidden_units = 512 # alias = C
    num_blocks = 3 # number of encoder/decoder blocks
    encoder_num_blocks = 6 # number of encoder blocks, will cover num_blocks. 0 will unused this
    decoder_num_blocks = 3 # number of decoder blocks, will cover num_blocks. 0 will unused this
    num_epochs = 100
    num_heads = 8
    dropout_rate = 0.1
    sinusoid = False # If True, use sinusoid. If false, positional embedding.
    
    
    
    
