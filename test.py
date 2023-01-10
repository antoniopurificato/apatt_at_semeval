import pickle
import pandas as pd

def tuple_to_list(my_tuple):
  my_list = []
  for name in my_tuple:
    my_list.append(name)
  output = ','.join(my_list).replace("'","")
  return output

with open('lightning_logs/Bert_ru_10_dictionary.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)
result = loaded_dict

my_list = []
for b,c in zip(result.keys(),result.values()):
  if b[2] == 0.2:
    my_list.append([b[0],b[1],tuple_to_list(c)])

my_df = pd.DataFrame(my_list, columns= ['Article_id','Line_id','Techniques'])

my_df.to_csv('Bert_ru_10_output.txt',header=None, index=None, sep='\t')