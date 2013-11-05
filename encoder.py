#!/usr/bin/env python
# coding: utf-8
# Author: Vladimir M. Zaytsev <zaytsev@usc.edu>

"""
This module contains implementations of various routines for
encoding and decoding raw data into feature vectors.
"""

S_BEGIN = "<<BEGIN>>" #chr(244)
S_END   = "<<END>>"   #chr(243)
U_WORD  = "<<UWORD>>" #chr(242)

WORD_TAG_SEP = "_"


class Token(object):

    def __init__(self, tag_prev, tag, word, sent_length, position, token_prev, unknown=False):
        self.tag = tag
        self.tag_prev = tag_prev
        self.raw = word
        self.token_prev = token_prev

        if word is not None:
            self.title = word.istitle()
            self.upper = word.isupper()
            self.form = word.lower()
            self.lemma = word.lower()
            self.word_len = len(word)
        else:
            self.title = False
            self.upper = False
            self.form = None
            self.lemma = None
            self.word_len = 0

        if unknown:
            self.form = U_WORD
            self.lemma = U_WORD
        self.position = position
        self.sent_len = sent_length
        self.norm_position = position

    def __repr__(self):
        return "<Token(%s, %s, %s, pos=%.3f)>" % (self.tag_prev,
                                                  self.tag,
                                                  self.form,
                                                  self.norm_position)


def tag(tagged_word):
    return tagged_word[1]


def word(tagged_word):
    return tagged_word[0]


def bigrams(sequence):
    seq_bigrams = []
    for i in xrange(len(sequence) - 1):
        seq_bigrams.append((sequence[i], sequence[i + 1]))
    return seq_bigrams


def words(tagged_words):
    return [tw[0] for tw in tagged_words]


def tags(tagged_words):
    return [tw[1] for tw in tagged_words]


class InputTransformer(object):

    def __init__(self):
        self.words = set()
        self.tags = set()

    def annotate(self, input_stream, tagged=True):
        annotated = []
        for line in input_stream:
            tagged_words = line.split()
            sent_len = len(tagged_words)
            tag_prev = S_BEGIN
            token_prev = None
            tokens = []
            for i, word_tag in enumerate(tagged_words):

                if tagged:
                    word, tag = word_tag.split(WORD_TAG_SEP)
                    token = Token(tag_prev, tag, word, sent_len, i, token_prev)
                    tag_prev = tag
                    self.words.add(token.form)
                    self.tags.add(token.tag)
                else:
                    word = word_tag
                    if word.lower() in self.words:
                        token = Token(tag_prev, None, word, sent_len, i, token_prev)
                    else:
                        token = Token(tag_prev, None, word, sent_len, i, token_prev, unknown=True)

                tokens.append(token)
                token_prev = token

            tokens.append(Token(tag_prev, S_END, None, sent_len, sent_len, token_prev))
            annotated.append(tokens)
        return annotated