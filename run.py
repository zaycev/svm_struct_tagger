#!/usr/bin/env python
# coding: utf-8
# Author: Vladimir M. Zaytsev <zaytsev@usc.edu>

import os
import sys
import pickle
import encoder
import features

from svm import SvmSolver


C = [0.1]
T = [0.1]
MAX_I = 3000


TRAIN_FILE   = "train.tags"
OUTPUT_FILE  = "output_dev_asgd3000.tags"
TEST_FILE    = "dev"
GOLD_FILE    = "dev.tags"


for c in C:
    for t in T:

        # 1. Define featureset
        feature_set = [
            features.F_T_WW(),
            features.F_T_W(),
            features.F_T_T(),
            features.F_T_C(),
            features.F_T_P(),
            features.F_T_SUF(),
            features.F_T_WORDLEN(),
            features.F_T_POSITION(),
            features.F_T_PUNCT(),
            features.F_T_DIGIT(),
        ]
        
        # Read data
        input_fl = open(TRAIN_FILE, "rb")
        transformer = encoder.InputTransformer()
        train = transformer.annotate(input_fl)
        input_fl.close()
        tag_set = list(transformer.tags)
        
        # Learn explicit features
        features.learn_feature_dicts(feature_set, train)
        
        # Train model `svm.fit` for OGD / `svm.ada_fit` for AOGD
        svm = SvmSolver()
        svm.ada_fit(train, feature_set, tag_set, C=c, theta=t, iters=MAX_I)
        
        # Open Dev/Test data
        dev_fl = open(TEST_FILE, "rb")
        dev = transformer.annotate(dev_fl, tagged=TRAIN_FILE==TEST_FILE)
        dev_fl.close()

        # Predict labels
        dev_tagged = svm.predict(dev)
        
        
        # Save predictions
        output_fl = open(OUTPUT_FILE, "wb")
        for tokens in dev_tagged:
            output_fl.write(" ".join([encoder.WORD_TAG_SEP.join((token.raw, token.tag))
                                      for token in tokens]))
            output_fl.write("\n")
        output_fl.close()

        # Print model and parameters
        print "C=%f, Theta=%f, I=%d" % (c, t, MAX_I)
        for f in svm.feature_set:
            print f
        
        # Print accuracy
        os.system("pypy eval.py %s %s" % (OUTPUT_FILE, GOLD_FILE))

        # Save model
        pickle.dump({"svm": svm, "transformer": transformer}, open("model_agd3000_dev.pkl", "wb"))
