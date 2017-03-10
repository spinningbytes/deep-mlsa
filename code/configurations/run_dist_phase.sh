#!/usr/bin/env bash
python runner.py -c mlsa_train/config_distant_phase_de.json
python runner.py -c mlsa_train/config_distant_phase_en.json
python runner.py -c mlsa_train/config_distant_phase_fr.json
python runner.py -c mlsa_train/config_distant_phase_it.json 