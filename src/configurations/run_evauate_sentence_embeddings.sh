#!/usr/bin/env bash
python runner.py -c mlsa_evaluate/config_supervised_de_sentence_embeddings.json 
python runner.py -c mlsa_evaluate/config_supervised_en_sentence_embeddings.json 
python runner.py -c mlsa_evaluate/config_supervised_fr_sentence_embeddings.json 
python runner.py -c mlsa_evaluate/config_supervised_it_sentence_embeddings.json 
