import torch
import torchvision
import numpy as np
import cv2

im=np.ones((1,3,224,224))
im_tensor = torch.FloatTensor(im)
model = torchvision.models.densenet121(pretrained=True)
feat = model.features(im_tensor)#shape:([1, 1024, 7, 7])
print (feat.shape)