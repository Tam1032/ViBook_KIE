import re
import json
import pandas as pd
import os
import py_vncorenlp

def getImagelabel(name_image, label_lines):
  #print(name_image)
  image_label= ''
  for line in label_lines:
    if name_image in line:
      image_label = line
      break
  jsonStr = json.loads(image_label.split('\t')[1])
  df = pd.DataFrame.from_dict(jsonStr)
  df[['text','label', 'link']]=df.transcription.str.rsplit('/',n=2, expand=True)
  df.sort_values(['label', 'link'],ascending = [False, True], inplace = True)
  df.reset_index(drop=True, inplace=True)
  df.loc[:,'image'] = name_image
  return  df

def load_file(label_file_path):
  fail = []
  with open(label_file_path, encoding="utf8") as file:
      lines = [line.strip() for line in file]
  img_list = [re.search("image_[0-9]+\.jpg", line).group() for line in lines]
  img_list = sorted(img_list)
  df = pd.DataFrame()
  for img in img_list:
      try:
        df = pd.concat([df, getImagelabel(img, lines)], axis=0, ignore_index=True)
      except:
        fail.append(img)
  df.drop(['difficult'], axis=1, inplace=True)
  return df, fail

def word_segment():
    py_vncorenlp.download_model()
    segmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"])
    return segmenter

def segment_words(df, segmenter):
  data = df.copy()
  length = data.shape[0]
  for i in range(length):
    data.loc[i, 'text'] = ' '.join(segmenter.word_segment(data.loc[i, 'text']))
  return data
