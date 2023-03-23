## Setup
This code can be runned only on a pc with a GPU and with `Python 3.8`.

Open a terminal.

You need first to create a `conda` environment:
```
conda create -n test_env python=3.8 && conda activate test_env && conda install pip
```

Clone this repository
```
git clone https://github.com/antoniopurificato/apatt_at_semeval.git
```

Move the current directory to this repository and install the required packages:
```
cd apatt_at_semeval && pip install -r requirements.txt
```


Download the dataset. The dataset is private and could be accessed only after a request to Semeval 2023 Task 3 organizers from this [site](https://propaganda.math.unipd.it/semeval2023task3/). Once you have the dataset run the following command:
```
tar -xzvf semeval2023task3bundle-v4.tgz
```

Then you have to create two folders required for the execution of the code:
```
mkdir lightning_logs && cd ..
```

After this you have to save the path of the dataset folder, it will be useful later!
```
cd semeval2023task3bundle-v4 && pwd && cd ..
```

Now you can start training your first model! You have first to train every single model and then you can create an ensemble, so you have to use `train_single_model.py`.
```
python3 train_single_model.py --epochs [NUMBER OF EPOCHS] --model [NAME OF THE MODEL] --language [LANGUAGE] --threshold [THRESHOLD VALUE EXPLAINED IN THE PAPER] --save [SAVE OR NOT THE MODEL] --path [PATH OF THE DATASET FOLDER (OUTPUT OF pwd IN THE PREVIOUS STEP)]
```

For other information, as the name of the avalaible models you can use:
```
python3 train_single_model.py --help
```

Once you have all the models for a language you can run the ensemble (make sure to have the right paths on the script `train_ensemble.py`).
```
python3 train_ensemble.py --epochs [NUMBER OF EPOCHS] --language [LANGUAGE] --threshold [THRESHOLD VALUE EXPLAINED IN THE PAPER] --save [SAVE OR NOT THE MODEL] --path [PATH OF THE DATASET FOLDER (OUTPUT OF pwd IN THE PREVIOUS STEP)]
```

For other information, as the name of the avalaible models you can use:
```
python3 train_ensmble.py --help
```

You can also test the results on the validation dataset!
```
python3 test.py --language [LANGUAGE] --threshold [THRESHOLD VALUE EXPLAINED IN THE PAPER] --model [NAME OF THE MODEL] --ensemble [IS AN ENSEMBLE OR NOT]
```
