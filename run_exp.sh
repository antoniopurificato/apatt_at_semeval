#!/bin/bash
python3 val.py --models Bert --epochs 10 --language en --threshold 0.1 --mode offline
python3 val.py --models Bert --epochs 10 --language it --threshold 0.1 --mode offline
python3 val.py --models Bert --epochs 10 --language ru --threshold 0.1 --mode offline
python3 val.py --models Bert --epochs 10 --language po --threshold 0.1 --mode offline
python3 val.py --models Bert --epochs 10 --language fr --threshold 0.1 --mode offline
python3 val.py --models Bert --epochs 10 --language ge --threshold 0.1 --mode offline
python3 val.py --models RoBERTa --epochs 10 --language en --threshold 0.1 --mode offline
python3 val.py --models RoBERTa --epochs 10 --language po --threshold 0.1 --mode offline
python3 val.py --models RoBERTa --epochs 10 --language ru --threshold 0.1 --mode offline
python3 val.py --models XLNet --epochs 10 --language en --threshold 0.1 --mode offline
python3 val.py --models DeBERTa --epochs 10 --language en --threshold 0.1 --mode offline
python3 val.py --models alBERT --epochs 10 --language en --threshold 0.1 --mode offline