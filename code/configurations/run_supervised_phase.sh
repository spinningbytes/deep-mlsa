#!/usr/bin/env bash
python runner.py -c mlsa_train/config_supervised_de.json 
python runner.py -c mlsa_train/config_supervised_en.json 
python runner.py -c mlsa_train/config_supervised_fr.json 
python runner.py -c mlsa_train/config_supervised_it.json 
