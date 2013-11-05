#!/usr/bin/env python
# coding: utf-8
# Author: Vladimir M. Zaytsev <zaytsev@usc.edu>

"""
Module containing diffirent feature functions.
"""

import re
import settings
import encoder
import collections

import string
import datrie

try:
    import numpy as np
except:
    import numpypy as np


WT_SEP  = "_" #chr(241)


def learn_feature_dicts(features, annotated_sents):
    for feature in features:
        feature.learn_dict(annotated_sents)

class Feature(object):

    def __init__(self):
        self.enc_dict = dict()
        self.dec_dict = dict()
        self.w = None

    def init_w(self, value=0):
        self.w = np.zeros(self.size(), dtype=settings.XDTYPE)
        self.w.fill(value)

    def size(self):
        # print len(self.enc_dict), self.enc_dict.prefix_items(u'a')
        return len(self.enc_dict)

    def learn_dict(self, tagged_sentences):
        raise NotImplementedError("Feature learn_dict")

    def encode(self, token):
        raise NotImplementedError("Feature encode")

    def score(self, token):
        score = settings.XDTYPE(0)
        for key_id, value in self.encode(token):
            score += value * self.w[key_id]
        return score

    def seq_score(self, tokens, tags):
        scores = collections.Counter()
        i = 0
        token = tokens[i]
        tag = tags[i]
        token.tag = tag
        token.tag_prev = encoder.S_BEGIN
        for key_id, value in self.encode(tokens[i]):
            scores[key_id] += value

        for i in xrange(1, len(tokens)):
            token = tokens[i]
            tag = tags[i]
            tag_prev = tokens[i - 1].tag
            token.tag = tag
            token.tag_prev = tag_prev
            for key_id, value in self.encode(tokens[i]):
                scores[key_id] += value

        return scores

    def update_weight(self, tokens, t_true, t_worst, theta, C):
        t_worst.append(encoder.S_END)
        worst_scores = self.seq_score(tokens, t_worst)
        true_scores = self.seq_score(tokens, t_true)
        changed_weigts = set(worst_scores.iterkeys())
        changed_weigts.update(true_scores.iterkeys())
        # updated = dict()
        for key_id in changed_weigts:
            w_reg = - theta * self.w[key_id] * C
            w_upd = + theta * (true_scores[key_id] - worst_scores[key_id])
            self.w[key_id] = self.w[key_id]  + w_reg + w_upd

            # updated[key_id] = self.w[key_id]  + w_upd + w_upd

        # self.decode(updated)
        # exit(0)
    
    def ada_update_weight(self, tokens, t_true, t_worst, g, C):
        t_worst.append(encoder.S_END)
        worst_scores = self.seq_score(tokens, t_worst)
        true_scores = self.seq_score(tokens, t_true)
        changed_weigts = set(worst_scores.iterkeys())
        changed_weigts.update(true_scores.iterkeys())
        

        for key_id in changed_weigts:
            w_reg = - self.w[key_id] * C
            w_upd = + (true_scores[key_id] - worst_scores[key_id])
            theta_i = np.power(w_upd, 2)
            if not key_id in g:
                pass
            else:
                for g_s in g[key_id]:
                    theta_i += np.power(g_s, 2)
            if theta_i <= 0:
                continue
            theta_i = np.sqrt(2) / np.sqrt(theta_i)
            self.w[key_id] = self.w[key_id]  + theta_i * (w_reg +  w_upd)
            if not key_id in g:
                g[key_id] = []
            g[key_id].append(w_upd)

    def w_size(self):
        return np.sqrt(np.sum(np.power(self.w, 2)))

    def decode(self, scores):
        for f_id, f_val in scores.iteritems():
            f_key = self.dec_dict[f_id]
            print "%d\t%.4f\t%s" % (f_id, f_val, f_key)

    def summary(self):
        for f_id, f_key in self.dec_dict.iteritems():
            f_val = self.w[f_id]
            word, tag = f_key.split(WT_SEP)
            if f_val != 0:
                print "%d\t%.4f\t%s" % (f_id, f_val, word + "_" + tag)


class F_T_W(Feature):

    def learn_dict(self, annotated_sents):

        tags = set()
        words = set([encoder.U_WORD])

        for annotated_sent in annotated_sents:
            for token in annotated_sent:
                if token.form is None:
                    continue
                word, tag = token.form, token.tag
                tags.add(tag)
                words.add(word)

        for word in words:
            for tag in tags:
                key = WT_SEP.join((word, tag))
                if key in self.enc_dict:
                    continue
                else:
                    key_id = len(self.enc_dict)
                    self.enc_dict[key] = key_id
                    self.dec_dict[key_id] = key

    def encode(self, token):
        if token.form is None:
            return []
        word, tag = token.form, token.tag
        key = WT_SEP.join((word, tag))
        key_id = self.enc_dict[key]
        return [(key_id, 1)]


class F_T_T(Feature):

    def learn_dict(self, annotated_sents):

        tags = set()

        for annotated_sent in annotated_sents:
            for token in annotated_sent:
                if token.tag == encoder.S_BEGIN or token.tag == encoder.S_END:
                    continue
                tags.add(token.tag)
                # print annotated_sent

        for tag_1 in tags:
            for tag_2 in tags:
                key = WT_SEP.join((tag_1, tag_2))
                if key in self.enc_dict:
                    continue
                else:
                    key_id = len(self.enc_dict)
                    self.enc_dict[key] = key_id
                    self.dec_dict[key_id] = key

        for tag in tags:
            key = WT_SEP.join((encoder.S_BEGIN, tag))
            if key in self.enc_dict:
                continue
            else:
                key_id = len(self.enc_dict)
                self.enc_dict[key] = key_id
                self.dec_dict[key_id] = key

        for tag in tags:
            key = WT_SEP.join((tag, encoder.S_END))
            if key in self.enc_dict:
                continue
            else:
                key_id = len(self.enc_dict)
                self.enc_dict[key] = key_id
                self.dec_dict[key_id] = key

    def encode(self, token):
        tag_prev, tag = token.tag_prev, token.tag

        key = WT_SEP.join((tag_prev, tag))
        key_id = self.enc_dict[key]
        return [(key_id, 1)]



class F_T_C(Feature):

    def learn_dict(self, annotated_sents):

        tags = set()

        for annotated_sent in annotated_sents:
            for token in annotated_sent:
                if token.form is None:
                    continue
                tags.add(token.tag)

        for case in ["u", "l", "t"]:
            for tag in tags:
                key = WT_SEP.join((case, tag))
                if key in self.enc_dict:
                    continue
                else:
                    key_id = len(self.enc_dict)
                    self.enc_dict[key] = key_id
                    self.dec_dict[key_id] = key

    def encode(self, token):
        if token.form is None:
            return []

        if token.upper:
            case = "u"
        elif token.title:
            case = "t"
        else:
            case = "l"

        key = WT_SEP.join((case, token.tag))
        key_id = self.enc_dict[key]
        return [(key_id, 1)]



class F_T_P(Feature):
    MINLEN = 4
    PLEN = 2

    def learn_dict(self, annotated_sents):

        tags = set()
        postfixes = set()

        for annotated_sent in annotated_sents:
            for token in annotated_sent:
                if token.form is None or len(token.form) < self.MINLEN:
                    continue
                word, tag = token.form, token.tag
                postfix = word[(len(word)-self.PLEN):]
                tags.add(tag)
                postfixes.add(postfix)

        for postfix in postfixes:
            for tag in tags:
                key = WT_SEP.join((postfix, tag))
                if key in self.enc_dict:
                    continue
                else:
                    key_id = len(self.enc_dict)
                    self.enc_dict[key] = key_id
                    self.dec_dict[key_id] = key

    def encode(self, token):
        if token.form is None or len(token.form) < self.MINLEN:
            return []
        word, tag = token.form, token.tag
        postfix = word[(len(word)-self.PLEN):]
        key = WT_SEP.join((postfix, tag))
        key_id = self.enc_dict.get(key)
        if key_id is not None:
            return [(key_id, 1)]
        return []

class F_T_POSITION(Feature):

    def learn_dict(self, annotated_sents):

        tags = set()

        for annotated_sent in annotated_sents:
            for token in annotated_sent:
                if token.form is None:
                    continue
                tags.add(token.tag)

        for position in ["f", "l", "?"]:
            for tag in tags:
                key = WT_SEP.join((position, tag))
                if key in self.enc_dict:
                    continue
                else:
                    key_id = len(self.enc_dict)
                    self.enc_dict[key] = key_id
                    self.dec_dict[key_id] = key

    def encode(self, token):
        if token.form is None:
            return []
        if token.position == 0:
            position = "f"
        elif token.position == token.sent_len - 1:
            position = "l"
        else:
            position = "?"
        key = WT_SEP.join((position, token.tag))
        key_id = self.enc_dict.get(key)
        if key_id is None:
            return []
        return [(key_id, 1)]


class F_T_SENTLEN(Feature):

    def learn_dict(self, annotated_sents):

        tags = set()

        for annotated_sent in annotated_sents:
            for token in annotated_sent:
                if token.form is None:
                    continue
                tags.add(token.tag)

        for sent_len in range(1, 32):
            for tag in tags:
                key = WT_SEP.join((str(sent_len), tag))
                if key in self.enc_dict:
                    continue
                else:
                    key_id = len(self.enc_dict)
                    self.enc_dict[key] = key_id
                    self.dec_dict[key_id] = key

    def encode(self, token):
        if token.form is None:
            return []
        sent_len = str(token.sent_len)
        key = WT_SEP.join((sent_len, token.tag))
        key_id = self.enc_dict.get(key)
        if key_id is None:
            return []
        return [(key_id, 1)]


class F_T_WORDLEN(Feature):

    def learn_dict(self, annotated_sents):

        tags = set()

        for annotated_sent in annotated_sents:
            for token in annotated_sent:
                if token.form is None:
                    continue
                tags.add(token.tag)

        for word_len in range(1, 32):
            for tag in tags:
                key = WT_SEP.join((str(word_len), tag))
                if key in self.enc_dict:
                    continue
                else:
                    key_id = len(self.enc_dict)
                    self.enc_dict[key] = key_id
                    self.dec_dict[key_id] = key

    def encode(self, token):
        if token.form is None:
            return []
        word_len = str(token.word_len)
        key = WT_SEP.join((word_len, token.tag))
        key_id = self.enc_dict.get(key)
        if key_id is None:
            return []
        return [(key_id, 1)]


class F_T_DIGIT(Feature):

    def learn_dict(self, annotated_sents):

        tags = set()

        for annotated_sent in annotated_sents:
            for token in annotated_sent:
                if token.form is None:
                    continue
                tags.add(token.tag)

        for digit_case in ["no", "has", "start", "all"]:
            for tag in tags:
                key = WT_SEP.join((digit_case , tag))
                if key in self.enc_dict:
                    continue
                else:
                    key_id = len(self.enc_dict)
                    self.enc_dict[key] = key_id
                    self.dec_dict[key_id] = key

    def encode(self, token):
        if token.form is None:
            return []
        digits = [ch.isdigit() for ch in token.form]
        if all(digits):
            d_case = "all"
        elif digits[0]:
            d_case = "start"
        elif any(digits):
            d_case = "has"
        else:
            d_case = "no"
        key = WT_SEP.join((d_case, token.tag))
        key_id = self.enc_dict.get(key)
        if key_id is None:
            return []
        return [(key_id, 1)]



class F_T_PUNCT(Feature):
    PUNCT = ".,-'\/"

    def learn_dict(self, annotated_sents):

        tags = set()

        for annotated_sent in annotated_sents:
            for token in annotated_sent:
                if token.form is None:
                    continue
                tags.add(token.tag)

        for punct_case in self.PUNCT:
            for tag in tags:
                key = WT_SEP.join((punct_case , tag))
                if key in self.enc_dict:
                    continue
                else:
                    key_id = len(self.enc_dict)
                    self.enc_dict[key] = key_id
                    self.dec_dict[key_id] = key

    def encode(self, token):
        if token.form is None:
            return []
        res = []
        for p in self.PUNCT:
            if p in token.form:
                key = WT_SEP.join((p, token.tag))
                key_id = self.enc_dict.get(key)
                if key_id is not None:
                    res.append((key_id, 1))
        return res



class F_T_SUF(Feature):
    MINLEN = 4
    PLEN = 2

    def learn_dict(self, annotated_sents):

        tags = set()
        postfixes = set()

        for annotated_sent in annotated_sents:
            for token in annotated_sent:
                if token.form is None or len(token.form) < self.MINLEN:
                    continue
                word, tag = token.form, token.tag
                postfix = word[:self.PLEN]
                tags.add(tag)
                postfixes.add(postfix)

        for postfix in postfixes:
            for tag in tags:
                key = WT_SEP.join((postfix, tag))
                if key in self.enc_dict:
                    continue
                else:
                    key_id = len(self.enc_dict)
                    self.enc_dict[key] = key_id
                    self.dec_dict[key_id] = key

    def encode(self, token):
        if token.form is None or len(token.form) < self.MINLEN:
            return []
        word, tag = token.form, token.tag
        postfix = word[:self.PLEN]
        key = WT_SEP.join((postfix, tag))
        key_id = self.enc_dict.get(key)
        if key_id is not None:
            return [(key_id, 1)]
        return []



class F_T_WW(Feature):

    def learn_dict(self, annotated_sents):

        tags = set()
        word_pairs = set()
        
        self.enc_dict = dict()#datrie.Trie(string.ascii_lowercase)
        self.dec_dict = None

        for annotated_sent in annotated_sents:
            for token in annotated_sent:
                if token.form is None:
                    continue

                tag = token.form
                tags.add(tag)

                if token.token_prev is None \
                or token.token_prev.form is None:
                    continue
                word_1 = token.token_prev.form
                word_2 = token.form
                
                key = unicode(WT_SEP.join((word_1, word_2, tag)))

                if key in self.enc_dict:
                    continue
                else:
                    key_id = len(self.enc_dict)
                    self.enc_dict[key] = key_id

    def encode(self, token):
        if token.form is None \
        or token.token_prev is None \
        or token.token_prev.form is None:
            return []
        word_1, word_2, tag = token.token_prev.form, token.form, token.tag
        key = unicode(WT_SEP.join((word_1, word_2, tag)))
        if key not in self.enc_dict:
            return []
        key_id = self.enc_dict[key]
        return [(key_id, 1)]
