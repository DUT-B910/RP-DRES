import config
import models
import numpy as np
import os
import time
import datetime
import json
from sklearn.metrics import average_precision_score
import sys
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type = str, default = 'GCN_node_all_path', help = 'name of the model')
parser.add_argument('--save_name', type = str, default = 'GCN_node_all_path_node_edge_node_checkpoint_avgpooling_demo')
parser.add_argument('--train_prefix', type = str, default = 'dev_train')
parser.add_argument('--test_prefix', type = str, default = 'dev_dev')
  
args = parser.parse_args()
model = {
	'GCGCN_node_path':models.GCGCN_node_path,
	'GCN_node_all_path':models.GCN_node_all_path,
}

con = config.Config(args)
con.set_max_epoch(100)
con.load_train_data()
con.load_test_data()
con.gen_train_facts_anno()
con.train(model[args.model_name], args.save_name)
