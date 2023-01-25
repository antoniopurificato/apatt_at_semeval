import pickle
import pandas as pd
import argparse
import subprocess
import re
import os

parser = argparse.ArgumentParser()
parser.add_argument('--threshold', type=float, help='Value of the threshold')
parser.add_argument('--language', type=str, help='Language')
parser.add_argument('--model', type=str, help='Model to be loaded')
parser.add_argument('--ensemble', type=str, default = 'False', help='Ensemble or not')
args = parser.parse_args()

def tuple_to_list(my_tuple):
  my_list = []
  for name in my_tuple:
    my_list.append(name)
  output = ','.join(my_list).replace("'","")
  return output
if args.ensemble == 'False':
  with open('lightning_logs/{}_{}_10_dictionary.pkl'.format(args.model,args.language), 'rb') as f:
    loaded_dict = pickle.load(f)
  result = loaded_dict
else:
  with open('lightning_logs/{}_ensemble_dictionary.pkl'.format(args.language), 'rb') as f:
      loaded_dict = pickle.load(f)
  result = loaded_dict

my_list = []
for b,c in zip(result.keys(),result.values()):
  if b[2] == args.threshold:
    my_list.append([b[0],b[1],tuple_to_list(c)])

my_df = pd.DataFrame(my_list, columns= ['Article_id','Line_id','Techniques'])

threshold_str = str(args.threshold).replace('.','')

if args.ensemble == 'False':
  my_df.to_csv('lightning_logs/test/test_{}_{}_{}.txt'.format(args.model,args.language,threshold_str),header=None, index=None, sep='\t')
else:
  my_df.to_csv('lightning_logs/test/test_ensemble_{}_{}.txt'.format(args.language,threshold_str),header=None, index=None, sep='\t')

if args.ensemble == 'False':
  bashCommand = "python3 semeval2023task3bundle-v3/scorers/scorer-subtask-3.py --techniques_file_path semeval2023task3bundle-v3/scorers/techniques_subtask3.txt  --pred_file_path lightning_logs/test/" + 'test_{}_{}_{}.txt'.format(args.model,args.language,threshold_str) + " --gold_file_path semeval2023task3bundle-v3/data/" + str(args.language) + "/dev-labels-subtask-3.txt"
else:
  bashCommand = "python3 semeval2023task3bundle-v3/scorers/scorer-subtask-3.py --techniques_file_path semeval2023task3bundle-v3/scorers/techniques_subtask3.txt  --pred_file_path lightning_logs/test/" + 'test_ensemble_{}_{}.txt'.format(args.language,threshold_str) + " --gold_file_path semeval2023task3bundle-v3/data/" + str(args.language) + "/dev-labels-subtask-3.txt"
process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
output, error = process.communicate()
mystr = str(output)
initial_string = mystr.split()[-1].replace('micro-F1=','').replace('macro-F1=','')
micro_f1, macro_f1 = re.findall(r"[-+]?(?:\d*\.*\d+)", initial_string)
if args.ensemble == 'False':
  my_list = [[args.model, args.language,args.threshold,micro_f1,macro_f1]]
else:
  my_list = [['Ensemble', args.language,args.threshold,micro_f1,macro_f1]]
my_df = pd.DataFrame(my_list, columns= ['Model','Language','Threshold','Micro F1', 'Macro F1'])
my_df.to_csv('lightning_logs/test/results.txt',mode='a', index=None,header = None, sep='\t')

os.chdir("lightning_logs/test")
if args.ensemble == 'False':
  bashCommand = "rm " + ' test_{}_{}_{}.txt'.format(args.model,args.language,threshold_str)
else:
  bashCommand = "rm " + ' test_ensemble_{}_{}.txt'.format(args.language,threshold_str)
process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
output, error = process.communicate()