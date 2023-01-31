#!/bin/bash
python3 prova.py --model Bert --language en --mode offline --save True --epochs 10 --threshold .2
python3 prova.py --model RoBERTa --language en --mode offline --save True --epochs 10 --threshold .2
python3 prova.py --model XLNet --language en --mode offline --save True --epochs 10 --threshold .2
python3 prova.py --model DeBERTa --language en --mode offline --save True --epochs 10 --threshold .2
python3 prova.py --model alBERT --language en --mode offline --save True --epochs 10 --threshold .2