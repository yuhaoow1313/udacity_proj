from utils import load_model, predict_function
import numpy as np
import json as js

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('img_path', type=str)
parser.add_argument('model_path', type=str)
parser.add_argument('--top_k', type=int)
parser.add_argument('--category_names', type=str, required=True)

args = parser.parse_args()
# print(args.img_path, args.model_path)

reloaded_keras_model = load_model(args.model_path)
prob, classes = predict_function(image_path=args.img_path, model=reloaded_keras_model, top_k=args.top_k)

with open(args.category_names, 'r') as f:
    class_names = js.load(f)
    final_classes= { flower_type_num: class_names[flower_type_num] for flower_type_num in classes }
    
print('Predictions:', final_classes, 
      '\nProbabilities:', prob)
