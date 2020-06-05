import cv2
import numpy as np
import numpy.random as rand
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import sys
from torchvision import transforms
from objectAttentionModelConvLSTM import *
from AttentionModelMS import *
from PIL import Image

class attentionMap(nn.Module):
  def __init__(self, backbone):
      super(attentionMap, self).__init__()
      self.backbone = backbone
      self.backbone.train(False)
      self.params = list(self.backbone.parameters())
      self.weight_softmax = self.params[-2]

  def forward(self, img_variable, img, size_upsample):
      logit, feature_conv, _ = self.backbone(img_variable)
      result = (logit, feature_conv, _)
      bz, nc, h, w = feature_conv.size()
      feature_conv = feature_conv.view(bz, nc, h*w)
      h_x = F.softmax(logit, dim=1).data
      probs, idx = h_x.sort(1, True)
      cam_img = torch.bmm(self.weight_softmax[idx[:, 0]].unsqueeze(1), feature_conv).squeeze(1)
      cam_img = F.softmax(cam_img, 1).data
      cam_img = cam_img.cpu().numpy()
      cam_img = cam_img.reshape(h, w)
      cam_img = cam_img - np.min(cam_img)
      cam_img = cam_img / np.max(cam_img)
      cam_img = np.uint8(255 * cam_img)
      output_cam = cv2.resize(cam_img, size_upsample)
      img = cv2.cvtColor(np.uint8(img), cv2.COLOR_RGB2BGR)
      heatmap = cv2.applyColorMap(output_cam, cv2.COLORMAP_JET)
      result = heatmap * 0.3 + img * 0.5
      return result



def genCAMS(rgbModel, rgbModel_MS, DATA_DIR, n_videos = 5, num_classes = 61, mem_size = 512):

  FRAME_FOLDER = "processed_frames2"
  CAM_FOLDER = "../Gen_CAMS"
  
  model1 = attentionModel(num_classes = num_classes, mem_size = mem_size)
  model2 = AttentionModelMS(num_classes = num_classes, mem_size = mem_size)
  
  model1.load_state_dict(torch.load(rgbModel))
  model2.load_state_dict(torch.load(rgbModel_MS))
  
  model1_backbone = model1.resNet
  model2_backbone = model2.resNet

  #we istantiate attentionMapModel using the backbone of already trained stage2 rgb model
  attentionMapModel = attentionMap(model1_backbone).cuda()
  attentionMapModel.train(False)
  for params in attentionMapModel.parameters():
    params.requires_grad = False

  #we istantiate attentionMapModel_MS using the backbone of already trained stage2 rgb model_MS
  attentionMapModel_MS = attentionMap(model2_backbone).cuda()
  attentionMapModel_MS.train(False)
  for params in attentionMapModel_MS.parameters():
    params.requires_grad = False
  
  rgb_dir = os.path.join(DATA_DIR, FRAME_FOLDER)
  subfolders = os.listdir(rgb_dir)
  
  if len(subfolders) != 4:
    raise FileNotFoundError("you specified the wrong directory")

  #generate the folder
  os.mkdir("../Gen_CAMS")
  users = []
  actions = []
  images = []

  #select n_videos random users
  for i in range(n_videos):
    users.append(subfolders[rand.randint(len(subfolders))])

  for user in users:
    user_dir = os.path.join(rgb_dir, user)
    
    #we select a random action
    acts = os.listdir(user_dir)
    action = acts[rand.randint(len(acts))]
    actions.append(action)

    frame_dir = os.path.join(user_dir, action, "1", "rgb")
    frames = np.array(sorted(os.listdir(frame_dir)))
    select_indices = np.linspace(0, len(frames), 5, endpoint=False, dtype=int)
    select_frames = frames[select_indices]
    selected_images = [os.path.join(frame_dir, frame) for frame in select_frames]
    images.append(selected_images)

  #use default transfomations
  normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
   )
  preprocess1 = transforms.Compose([
    transforms.Scale(256),
    transforms.CenterCrop(224),
  ])

  preprocess2 = transforms.Compose([
      transforms.ToTensor(),
      normalize])

  #cicle all images
  for frames, user, action in zip(images, users, actions):
    path_folder = os.path.join(CAM_FOLDER, f"{user}_{action}")
    os.mkdir(path_folder)
    for frame in frames:
      
      filename = os.path.split(frame)[-1].split(".")[0]
      
      fl_nocam = os.path.join(path_folder, f"{filename}.png")
      fl_cam = os.path.join(path_folder, f"{filename}_CAM.png")
      fl_cam_MS = os.path.join(path_folder, f"{filename}_CAM_MS.png")
      img_pil = Image.open(frame)
      img_pil1 = preprocess1(img_pil)
      size_upsample = img_pil1.size
      
      #save the original image
      img_pil1.save(fl_nocam)

      img_tensor = preprocess2(img_pil1) #3*224*224
      img_variable = Variable(img_tensor.unsqueeze(0).cuda()) #1*3*224*224
      img = np.asarray(img_pil1)
      attentionMap_image = attentionMapModel(img_variable, img, size_upsample)
      attentionMap_image_MS = attentionMapModel_MS(img_variable, img, size_upsample)
      
      #save the new images
      cv2.imwrite(fl_cam, attentionMap_image)
      cv2.imwrite(fl_cam_MS, attentionMap_image_MS)
