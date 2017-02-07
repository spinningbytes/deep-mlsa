# -*- coding: utf-8 -*-
from nltk.tokenize import TweetTokenizer, WordPunctTokenizer
import re

class Preprocessing(object):
    def __init__(self, types):
        self.preprocessors = {}
        for t in types:
            if t == 'tweets':
                self.preprocessors[t] = TwitterPreprocessing()

    def preprocess_text(self, text, text_type, **kwargs):
        return self.preprocessors[text_type].preprocess(text, kwargs.get('language', 'default'))


class TwitterPreprocessing(object):
    def __init__(self):
        self.tokenizers = {
            'en': TweetTokenizer(),
            'de': WordPunctTokenizer(),
            'it': WordPunctTokenizer(),
            'fr': WordPunctTokenizer(),
            'default': WordPunctTokenizer()
        }

        self.tokenizer = TweetTokenizer()

    def preprocess(self, tweet, lang):
        #lowercase and normalize urls
        tweet = tweet.lower()
        tweet = tweet.replace('\n', '')
        tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))', 'URLTOK', tweet)
        tweet = re.sub('@[^\s]+', 'USRTOK', tweet)
        tweet = re.sub(r'#([^\s]+)', r'\1', tweet)

        tweet = self.tokenizers.get(lang, self.tokenizer).tokenize(tweet)

        return list(map(lambda x: x.replace(' ', ''), tweet))


