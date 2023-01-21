import pickle
import pandas as pd
import argparse
import subprocess
import re
import os


def tuple_to_list(my_tuple):
  my_list = []
  for name in my_tuple:
    my_list.append(name)
  output = ','.join(my_list).replace("'","")
  return output

languages = ['en', 'it', 'ge', 'po', 'fr', 'ru']
models = ['Bert', 'RoBERTa', 'XLNet', 'DeBERTa', 'alBERT']
thresholds = [.1,.2,.3,.4]
for language in languages:
  for model in models:
    for threshold in thresholds:
      if os.path.exists('lightning_logs/{}_{}_10_dictionary.pkl'.format(model,language)):
        with open('lightning_logs/{}_{}_10_dictionary.pkl'.format(model,language), 'rb') as f:#(args.model,args.language), 'rb') as f:
            loaded_dict = pickle.load(f)
        result = loaded_dict

        my_list = []
        for b,c in zip(result.keys(),result.values()):
          if b[2] == threshold:#args.threshold:
            my_list.append([b[0],b[1],tuple_to_list(c)])

        my_df = pd.DataFrame(my_list, columns= ['Article_id','Line_id','Techniques'])

        threshold_str = str(threshold).replace('.','')#str(args.threshold).replace('.','')

        my_df.to_csv('lightning_logs/test/test_{}_{}_{}.txt'.format(model,language,threshold_str),header=None, index=None, sep='\t') #format(args.model,args.language,threshold_str)

        bashCommand = "python3 semeval2023task3bundle-v3/scorers/scorer-subtask-3.py --techniques_file_path semeval2023task3bundle-v3/scorers/techniques_subtask3.txt  --pred_file_path lightning_logs/test/" + 'test_{}_{}_{}.txt'.format(model,language,threshold_str) + " --gold_file_path semeval2023task3bundle-v3/data/" + str(language) + "/dev-labels-subtask-3.txt" #(args.model,args.language,threshold_str) && args.language
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        mystr = str(output)
        initial_string = mystr.split()[-1].replace('micro-F1=','').replace('macro-F1=','')
        micro_f1, macro_f1 = re.findall(r"[-+]?(?:\d*\.*\d+)", initial_string)
        my_list = [[model, language, threshold, micro_f1, macro_f1]] #args.model, args.language,args.threshold,micro_f1,macro_f1
        my_df = pd.DataFrame(my_list, columns= ['Model','Language','Threshold','Micro F1', 'Macro F1'])
        my_df.to_csv('lightning_logs/test/results.txt',mode='a', index=None,header = None, sep='\t')

        os.chdir("lightning_logs/test")
        bashCommand = "rm " + ' test_{}_{}_{}.txt'.format(model,language,threshold_str) #format(args.model,args.language,threshold_str)
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        os.chdir("../..")