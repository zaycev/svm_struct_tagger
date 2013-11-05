#!/usr/bin/env python
# coding: utf-8
# Author: Vladimir M. Zaytsev <zaytsev@usc.edu>

import encoder
import settings
from random import choice

try:
    import numpy as np
except:
    import numpypy as np

XDTYPE = np.float64
YDTYPE = np.int16


class SvmSolver(object):

    def __init__(self):
        self.feature_set = None
        self.tag_set = None

    def total_local_score(self, token, feature_set):
        score = settings.XDTYPE(0)
        for f in feature_set:
            score += f.score(token)
        return score

    def find_worst_answer(self, tokens, true_answer, feature_set, tag_set):
        for t in tokens:
            t.tag = None
            t.tag_prev = None

        n = len(tokens)
        m = len(tag_set)
        F = np.zeros((n, m), dtype=settings.XDTYPE)
        B = np.zeros((n, m), dtype=np.int16)

        i = 0
        token = tokens[i]
        token.tag_prev = encoder.S_BEGIN

        for j, tag in enumerate(tag_set):
            token.tag = tag
            B[i, j] = j
            F[i, j] = self.total_local_score(token, feature_set)

            if true_answer[i] != tag:
                F[i, j] += 1

        for i in xrange(1, n - 1):
            token = tokens[i]

            for j in xrange(m):

                tag = tag_set[j]
                token.tag = tag
                best_score = -np.inf
                best_prev = None

                for k in xrange(m):
                    tag_prev = tag_set[k]
                    prev_score = F[i - 1, k]
                    token.tag_prev = tag_prev
                    score = self.total_local_score(token, feature_set)
                    score = prev_score + score
                    if tag != true_answer[i]:
                        score += 1

                    if score > best_score:
                        best_score = score
                        best_prev = k

                F[i, j] = best_score
                B[i, j] = best_prev

        i = n - 1
        best_score = -np.inf
        best_tag_i = 0
        token = tokens[i]
        for j in xrange(m):
            token.tag_prev = tag_set[j]
            token.tag = encoder.S_END
            score = self.total_local_score(token, feature_set)
            prev_score = F[i - 1, j]
            score = score + prev_score
            if score > best_score:
                best_tag_i = j
                best_score = score
            F[i, j] = score


        best_path = [best_tag_i]
        i = n - 2

        while i > 0:

            best_tag_i = B[i, best_tag_i]
            best_path.append(best_tag_i)
            i -= 1

        return map(tag_set.__getitem__, reversed(best_path))


    def fit(self, annotated_input, feature_set, tag_set, theta=XDTYPE(0.1), C=XDTYPE(0.1), iters=None):

        for f in feature_set:
            f.init_w(0.1)

        i = 0
        for i in xrange(iters):
            j = i % len(annotated_input)
            # tokens = annotated_input[j]
            tokens = choice(annotated_input)

            t_true = [token.tag for token in tokens]
            try:
                t_worst = self.find_worst_answer(tokens, t_true, feature_set, tag_set)
            except Exception:
                import traceback
                print tokens
                traceback.print_exc()
                exit(0)

            for feature in feature_set:
                feature.update_weight(tokens, t_true, t_worst, theta, C)


        self.feature_set = feature_set
        self.tag_set = tag_set

    def ada_fit(self, annotated_input, feature_set, tag_set, theta=XDTYPE(0.1), C=XDTYPE(0.1), iters=None):
        
        g = dict()
        
        for f in feature_set:
            f.init_w(0.1)

        for i in xrange(iters):
            tokens = choice(annotated_input)

            t_true = [token.tag for token in tokens]
            t_worst = self.find_worst_answer(tokens, t_true, feature_set, tag_set)

            for feature in feature_set:
                feature.ada_update_weight(tokens, t_true, t_worst, g, C)


        self.feature_set = feature_set
        self.tag_set = tag_set



    def find_best_answer(self, tokens, feature_set, tag_set):
        for t in tokens:
            t.tag = None
            t.tag_prev = None


        n = len(tokens)
        m = len(tag_set)
        F = np.zeros((n, m), dtype=settings.XDTYPE)
        B = np.zeros((n, m), dtype=np.int16)

        i = 0
        token = tokens[i]
        token.tag_prev = encoder.S_BEGIN

        for j, tag in enumerate(tag_set):
            token.tag = tag
            B[i, j] = j
            F[i, j] = self.total_local_score(token, feature_set)

        for i in xrange(1, n - 1):
            token = tokens[i]

            for j in xrange(m):

                tag = tag_set[j]
                token.tag = tag
                best_score = -np.inf
                best_prev = None

                for k in xrange(m):
                    tag_prev = tag_set[k]
                    prev_score = F[i - 1, k]
                    token.tag_prev = tag_prev
                    score = self.total_local_score(token, feature_set)


                    score = prev_score + score

                    if score > best_score:
                        best_score = score
                        best_prev = k

                F[i, j] = best_score
                B[i, j] = best_prev

        i = n - 1
        best_score = -np.inf
        best_tag_i = 0
        token = tokens[i]
        for j in xrange(m):
            token.tag_prev = tag_set[j]
            token.tag = encoder.S_END
            score = self.total_local_score(token, feature_set)
            prev_score = F[i - 1, j]
            score = score + prev_score
            if score > best_score:
                best_tag_i = j
                best_score = score
            F[i, j] = score


        best_path = [best_tag_i]
        i = n - 2

        while i > 0:

            best_tag_i = B[i, best_tag_i]
            best_path.append(best_tag_i)
            i -= 1

        return map(tag_set.__getitem__, reversed(best_path))

    def predict(self, annotated_input):

        for tokens_i, tokens in enumerate(annotated_input):

            try:
                best_path = self.find_best_answer(tokens, self.feature_set, self.tag_set)
            except Exception:
                import traceback
                print tokens
                print tokens_i
                traceback.print_exc()
                exit(0)


            prev_tag = encoder.S_BEGIN
            for i, tag in enumerate(best_path):
                tokens[i].tag = tag
                tokens[i].prev_tag = prev_tag
                prev_tag = tag

            yield tokens[:(len(tokens) - 1)]



