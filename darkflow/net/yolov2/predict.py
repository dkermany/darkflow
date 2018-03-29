import numpy as np
import math
import cv2
import os
import json
import matplotlib.pyplot as plt
#from scipy.special import expit
#from utils.box import BoundBox, box_iou, prob_compare
#from utils.box import prob_compare2, box_intersection
from collections import Counter
from ...utils.box2 import IOU
from ...cython_utils.cy_yolo2_findboxes import box_constructor
from copy import deepcopy
import line_profiler
from pprint import pprint

def expit(x):
  return 1. / (1. + np.exp(-x))

def _softmax(x):
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out

def findboxes(self, net_out, threshold):
  # meta
  meta = self.meta
  boxes = list()
  boxes = box_constructor(meta, threshold, net_out)
  return boxes

def postprocess_OCT(self, json, imgpath):
  """
  Takes json input and im path and saves jpeg with best selected thresholds
  """

  meta = self.meta
  colors = meta["colors"]
  labels = meta["labels"]
  assert type(imgpath) is str
  img = cv2.imread(imgpath)
  h, w, _ = img.shape
  thickness = int((h + w) // 2000)
  img_name = os.path.basename(imgpath).split(".")[0]
  
  thresholds = {
    "CNV": 0.2,
    "RF" : 0.2,
    "GA" : 0.15,
    "DRU": 0.20,
    "EX" : 0.15,
    "ERM": 0.25
  }

  relevant_boxes = [ bb for bb in json if bb["confidence"] > thresholds[bb["label"]]]
  current_img = np.copy(img)

  for bb in relevant_boxes:
    label = bb["label"]
    xmin, ymin = bb["topleft"]["x"], bb["topleft"]["y"]  
    xmax, ymax = bb["bottomright"]["x"], bb["bottomright"]["y"]

    # Draw bounding box and label
    cv2.rectangle(current_img, (xmin, ymin),(xmax, ymax),
      colors[labels.index(label)], thickness)
    # cv2.putText(current_img, label, (xmin, ymin - 12), 0, 1e-03 * h, 
    #   colors[labels.index(label)], thickness)

  # Threshold Label
  # cv2.putText(current_img, "threshold: {0:.1f}%".format(threshold * 100), (40, 100), 0, 3e-03 * h, 
    # (255,255,255), thickness*2)
  for j, label in enumerate(labels):
    cv2.putText(current_img, label, (100 + (200 * j), h - 40), 0, 1e-03 * h,
      colors[labels.index(label)], int(thickness*1.2))

  return current_img

#@profile
def postprocess_tif(self, json, imgpath):
  """
  Takes json input and im path and saves thresholded predictions
  to a tif stack for manual grading
  """
  # meta
  meta = self.meta
  colors = meta["colors"]
  labels = meta["labels"]
  assert type(imgpath) is str
  img = cv2.imread(imgpath)
  h, w, _ = img.shape
  thickness = int((h + w) // 2000)

  outfolder = os.path.join(self.FLAGS.imgdir, "out")
  if not os.path.exists(outfolder): os.makedirs(outfolder)
  tif_stack = np.zeros((20, h, w, 3), "uint8")

  for i, threshold in enumerate(np.arange(0.05, 1.05, 0.05)):
    relevant_boxes = [ bb for bb in json if bb["confidence"] > threshold ]
    current_img = np.copy(img)
    
    for bb in relevant_boxes:
      #if threshold > bb["confidence"]:
      #  continue

      label = bb["label"]
      xmin, ymin = bb["topleft"]["x"], bb["topleft"]["y"]
      xmax, ymax = bb["bottomright"]["x"], bb["bottomright"]["y"]
 
      # Draw bounding box and label
      cv2.rectangle(current_img, (xmin, ymin),(xmax, ymax),
        colors[labels.index(label)], thickness)
      cv2.putText(current_img, label, (xmin, ymin - 12), 0, 1e-03 * h, 
        colors[labels.index(label)], thickness)

    # Threshold Label
    cv2.putText(current_img, "threshold: {0:.1f}%".format(threshold * 100), (40, 100), 0, 3e-03 * h, 
      (255,255,255), thickness*2)
    # for j, label in enumerate(labels):
    #   cv2.putText(current_img, label, (100 + (200 * j), h - 40), 0, 1e-03 * h,
    #     colors[labels.index(label)], thickness*1.5)


    tif_stack[i] = current_img

  # assert tif_stack is not all zeros
  assert np.any(tif_stack)
  return tif_stack

   
def postprocess(self, net_out, im, threshold, save = True):
  """
  Takes net output, draw net_out, save to disk
  """
  inp_path = os.path.dirname(im)

  if self.FLAGS.evaluate or self.FLAGS.classify or self.FLAGS.json2tif or self.FLAGS.clinic:
    threshold = 0.0

  boxes = self.findboxes(net_out, threshold)

  # meta
  meta = self.meta
  colors = meta['colors']
  labels = meta['labels']
  if type(im) is not np.ndarray:
    imgcv = cv2.imread(im)
  else: imgcv = im
  h, w, _ = imgcv.shape
  
  resultsForJSON = []
  for b in boxes:
    boxResults = self.process_box(b, h, w, threshold)
    if boxResults is None:
      continue
    left, right, top, bot, mess, max_indx, confidence = boxResults
    thick = int((h + w) // 2000)
    if self.FLAGS.json or self.FLAGS.evaluate or self.FLAGS.classify or self.FLAGS.json2tif or self.FLAGS.clinic:
      resultsForJSON.append({
        "label": mess, 
        "confidence": float('%.2f' % confidence), 
        "topleft": {"x": left, "y": top}, 
        "bottomright": {"x": right, "y": bot}})
      if self.FLAGS.json:
        continue

    cv2.rectangle(imgcv,
      (left, top), (right, bot),
      colors[max_indx], thick)
    cv2.putText(imgcv, mess, (left, top - 12),
     0, 1e-3 * h, colors[max_indx], thick)

  if not save: return imgcv

  outfolder = os.path.join(inp_path, 'out')
  if not os.path.exists(outfolder): os.makedirs(outfolder)
  img_name = os.path.join(outfolder, os.path.basename(im))
  if self.FLAGS.json or self.FLAGS.evaluate or self.FLAGS.classify or self.FLAGS.json2tif or self.FLAGS.clinic:
    textJSON = json.dumps(resultsForJSON)
    textFile = os.path.splitext(img_name)[0] + ".json"
    with open(textFile, 'w') as f:
      f.write(textJSON)
    if self.FLAGS.json or self.FLAGS.clinic:
      return

  cv2.imwrite(img_name, imgcv)
