# -*- coding: utf-8 -*-
from nltk.tokenize import TweetTokenizer, WordPunctTokenizer
import re


tokenizers = {
    'en': TweetTokenizer(),
    'de': WordPunctTokenizer(),
    'it': WordPunctTokenizer(),
    'fr': WordPunctTokenizer(),
    'default': WordPunctTokenizer()
}

tokenizer = TweetTokenizer()


def preprocess(tweet, lang):
    #lowercase and normalize urls
    tweet = tweet.lower()
    tweet = tweet.replace('\n', '')
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))', 'URLTOK', tweet)
    tweet = re.sub('@[^\s]+', 'USRTOK', tweet)
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)

    tweet = tokenizers.get(lang, tokenizer).tokenize(tweet)

    return list(map(lambda x: x.replace(' ', ''), tweet))


