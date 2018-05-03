import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from classifier.simpleClassifier import simpleClassifier
from classifier.vggClassifier import vggClassifier
import os
from collections import OrderedDict
import torch 
from torch.autograd import Variable
import numpy as np

def load_model(model, path, name, mode = "parallel"):
	state_dict = torch.load(os.path.join(path, name))["model"]
	new_state_dict = OrderedDict()

	for k, v in state_dict.items():
		name = ""
		if mode == "single" and k.startswith("module."):
			name = k[7:]
		elif mode == "parallel" and not k.startswith("module."):
			name = "module."+k
		else:
			name = k
		new_state_dict[name] = v
		
	model.load_state_dict(new_state_dict)

def feat_ext(ext_type, cuda = True):
	temp_model = None
	model = None
	num_layers = 0

	if ext_type == "simpleClassifier42":
		temp_model = simpleClassifier()        
		load_model(temp_model, "/data/classifier/trainedModels/", "default_modelsimpleClassifier42.pth", mode="single")
		num_layers = 3

	elif ext_type == "simpleClassifier58":
		temp_model = simpleClassifier()        
		load_model(temp_model, "/data/classifier/trainedModels/", "default_modelsimpleClassifier58.pth", mode="single")
		num_layers = 3        
        
	elif ext_type == "vggClassifier67":
		temp_model = vggClassifier()        
		load_model(temp_model, "/data/classifier/trainedModels/", "default_modelvggClassifier67.pth", mode="single")
		num_layers = 3
        
	for param in temp_model.parameters():
		param.requires_grad = False

	model = torch.nn.Sequential(*(temp_model.features[i] for i in range(num_layers)))

	if cuda:
		model.cuda()

	return model

def applyFeatureTransformation(img, ext_type):
	feat_extractor = feat_ext(ext_type, cuda = True)
	img = img.astype(np.float32)
	img = np.expand_dims(img, axis=2)
	img = np.transpose(img, [2,0,1])
	img = np.expand_dims(img, axis=0)
	img = Variable(torch.from_numpy(img))
	img = img.cuda()
	feat_extractor = feat_ext(ext_type, cuda=True)
	feature_space_img = feat_extractor(img)
	feature_space_img = (feature_space_img.data).cpu().numpy()
	feature_space_img = feature_space_img[0,0,:,:]    
	return feature_space_img
