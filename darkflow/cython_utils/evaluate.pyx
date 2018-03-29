import os
import time
import numpy as np
cimport cython
import json
import matplotlib.pyplot as plt
from pprint import pprint
from glob import glob

"""
Boxes defined as a dict with keys
{xn, yn, xx, yx}
"""

# Intersection over Union using coordinates
cdef float IOU(dict A, dict B):
  return intersection(A, B) / float(union(A, B))

# Return Area if overlap
cdef float intersection(dict A, dict B):
  assert A['xx'] > A['xn'] and B['xx'] > B['xn'] and \
         A['yx'] > A['yn'] and B['yx'] > B['yn']

  cdef int w, h

  w = min(A['xx'], B['xx']) - max(A['xn'], B['xn'])
  h = min(A['yx'], B['yx']) - max(A['yn'], B['yn'])
  if w >= 0 and h >= 0:
    return w * h
  return 0.0

cdef float union(dict A, dict B):
  cdef float i = intersection(A, B)
  return area(A) + area(B) - i

cdef float area(dict A):
  cdef float area = (A['xx'] - A['xn']) * (A['yx'] - A['yn'])  
  assert area >= 0
  return area



cpdef cy_evaluate(dict meta, object framework, object FLAGS):
  cdef:
    list _labels = list()
    list gt_data = list()
    list image = list()
    list json_paths = list()
    list b = list()

    dict gt_boxes = dict()
    dict class_eval = dict()
    dict tally = dict()
    dict pred_feed = dict()
    dict gt_feed = dict()
    dict bb = dict()

    float threshold, confidence, iou
    float precision, recall

    int i, j, k, p
    int w, h
    int xmin, xmax, ymin, ymax
    int _xmin, _xmax, _ymin, _ymax
    int tp, fp, fn
    int N

    int M, P, Q
    list gt_labels = list()
    list gt_check = list()
    list gt_check_list = list()


    #str current_label, jsfilename, image_filename, _label


  _labels = meta["labels"]
  print("cython yo")

  # Load gt boxes
  gt_data = framework.parse()
  gt_boxes = { image[0]: image[1] for image in gt_data }

  # Load predicted boxes
  outfolder = os.path.join(FLAGS.imgdir, "out")
  json_paths = glob("{}/*.json".format(outfolder))
  assert len(json_paths) == len(gt_boxes.keys())

  # Loop through each of the objects in labels.txt
  for current_label in _labels:
    #print("\n{} predictions".format(current_label))
    class_eval = { "precision": [], "recall": [] }

    # Loop through all thresholds
    for threshold in np.arange(0.00, 1.00, 0.05):
      threshold = min(threshold, 1.00)
      tally = { "tp": 0, "fp": 0, "fn": 0 }

      # Loop through predicted box outputs
      N = len(json_paths)
      for i in range(N):
        jsfilename = json_paths[i]

        image_filename = "{}.jpeg".format(os.path.basename(jsfilename).split(".")[0])
        assert image_filename in gt_boxes.keys()
        #print("Image #{}: {}".format(j, image_filename))

        # Extract gt_info
        w, h, current_gt = gt_boxes[image_filename]
      
        # replaces 
        # gt_labels = [ b[0] for b in current_gt ]
        # Create GT Variables for tracking
        # gt_check = [ False for b in current_gt if b[0] == current_label ]
        # gt_check_list = [ b for b in current_gt if b[0] == current_label ]
        
        M = len(current_gt)
        gt_check = list()
        gt_check_list = list()
        for k in range(M):
          gt_labels.append(current_gt[k][0])
          if current_gt[k][0] == current_label:
            gt_check.append(False)
            gt_check_list.append(current_gt[k])

      
        # Load json predictions as a list of dictionaries
        with open(jsfilename) as f:
          js = json.load(f)

        # Loop over each bounding box prediction
        P = len(js)
        for p in range(P):
          bb = js[p]

          label = bb["label"]
          confidence = bb["confidence"]
          # x increases to the right, y increases down
          xmin = bb["topleft"]["x"]
          ymin = bb["topleft"]["y"]
          xmax = bb["bottomright"]["x"]
          ymax = bb["bottomright"]["y"]

          if label != current_label: continue
          if confidence < threshold: continue

          # if predicted label is not 
          if label not in gt_labels:
            tally["fp"] += 1
            continue

          # Create dict to feed into IOU function
          pred_feed = { "xn": xmin, "xx": xmax, "yn": ymin, "yx": ymax }

          # Loop over each ground-truth box
          Q = len(gt_check_list)
          for j in range(Q):
            box_gt = gt_check_list[j]
            # label, left, top, right, bottom (ymax = bottom)
            _label = box_gt[0]
            _xmin = box_gt[1]
            _ymin = box_gt[2] 
            _xmax = box_gt[3]
            _ymax = box_gt[4]

            assert _label == current_label

            # Create dict to feed into IOU function
            gt_feed = { "xn": _xmin, "xx": _xmax, "yn": _ymin, "yx": _ymax }

            try:
              iou = IOU(pred_feed, gt_feed)
            except AssertionError:
              print("IOU Assertion Error")
              continue
           
            if iou >= 0.4 and not gt_check[j]:
              tally["tp"] += 1
              gt_check[j] = True
              break
            # if prediction matches ground-truth already discovered, ignore
            elif iou >= 0.4 and gt_check[j]:
              continue

        tally["fn"] += gt_check.count(False)

      #pprint(tally)
      # Calculate precision and recall at each threshold
      tp = tally["tp"]
      fp = tally["fp"]
      fn = tally["fn"]

      if tp == fp == fn == 0:
        #print("Warning: No predictions or ground-truth annotations")
        continue
      elif tp == fp == 0:
        precision = 1.0
        recall = 0.0
      elif tp == fn == 0:
        precision = 0.0
        recall = 1.0
      else:
        precision = float(tp) / float(tp + fp)
        recall = float(tp) / float(tp + fn)

      class_eval["precision"].append(precision)
      class_eval["recall"].append(recall)

    #pprint(class_eval)
    #plt.plot(class_eval["precision"], class_eval["recall"])
  #plt.show()