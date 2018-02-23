import numpy as np
import math
import cv2
import os
import json
#from scipy.special import expit
#from utils.box import BoundBox, box_iou, prob_compare
#from utils.box import prob_compare2, box_intersection
from ...utils.box2 import IOU
from ...cython_utils.cy_yolo2_findboxes import box_constructor

def expit(x):
	return 1. / (1. + np.exp(-x))

def _softmax(x):
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out

def findboxes(self, output_image):
	# meta
	meta = self.meta
	boxes = list()
	boxes = box_constructor(meta,output_image)
	return boxes

# Calculate precision, recall, mAP
def evaluate(self, output_batch, filenames, box_info, detection=False, redundant=False):
  # Meta
  _labels = self.meta["labels"]

  # Create dict to store precision/recall points
  PR = { _class: { "precision": [], "recall": [] } for _class in _labels }

  # Load ground-truth boxes into dict
  gt_boxes = { image[0]: image[1] for image in box_info }

  # Loops over thresholds
  for threshold in np.arange(1.00, 0.00, -0.05): 
    
    # Initialize logging dictionary
    if detection: oneclass_eval = { "TP": 0, "FP": 0, "FN": 0 }
    else: class_eval = { _class: { "TP": 0, "FP": 0, "FN": 0 } for _class in _labels } 

    # Loops over each image in output batch
    for i, output_image in enumerate(output_batch):
      current_filename = filenames[i]
      assert current_filename in gt_boxes.keys()

      npimg = cv2.imread(os.path.join(self.FLAGS.imgdir, current_filename))
      w, h, _ = npimg.shape
      #h, w, _ = npimg.shape

      # Extract gt_info
      gt_h, gt_w, current_gt = gt_boxes[current_filename]
      assert h == gt_h and w == gt_w   

      current_predicted = self.findboxes(output_image)

      # Loop over each class
      for current_class in _labels:

        # Create GT variables for evaluation
        gt_check = [ False for m in current_gt if m[0] == current_class ]
        current_gt_class = [ m for m in current_gt if m[0] == current_class ]

        # Loop over each predicted box
        for j, box_predicted in enumerate(current_predicted):
          print("{}    {curre}".format(current_class, j))
          # Variable to track prediction
          correct_prediction = False

          # Get predicted coordinates
          predicted_results = self.process_box(box_predicted, h, w, threshold)
          if predicted_results is None: continue
          left, right, bot, top, label, max_indx, confidence = predicted_results
          if label != current_class and not detection: continue

          # Reorganize into dict
          pred_dict = { "xn": left, "xx": right, "yn": bot, "yx": top }

          # Loop over each ground-truth box
          for k, box_gt in enumerate(current_gt_class):
   
            # if evaluation is not redundant and this box has been found
            if gt_check[k] and not redundant: continue

            # Get ground-truth coordinates and reorganize into dict
            _label, _left, _bot, _right, _top = box_gt
            gt_dict = { "xn": _left, "xx": _right, "yn": _bot, "yx": _top }
            assert _label == label

            # Calculate Intersection-Over-Union of current boxes
            iou = IOU(pred_dict, gt_dict)
            print("IOU: {}".format(iou))
            # Boxes overlap sufficiently (True Positives)
            if iou >= 0.4:
              correct_prediction = True
              gt_check[k] = True
              if detection: oneclass_eval["TP"] += 1
              else: class_eval[current_class]["TP"] += 1
              break

          # Prediction did not match any of the ground-truth boxes of this class
          if not correct_prediction: 
            class_eval[current_class]["FP"] += 1
            print("FALSE POSITIVE")
          
        # Ground-truth boxes NOT detected (False Negatives)
        if detection: oneclass_eval["FN"] += gt_check.count(False)
        else: class_eval[current_class]["FN"] += gt_check.count(False)
      
      # Summary for classes
      #for class_ in class_eval:
       # print("{}:    {} TP  {} FP  {} FN".format(
        #    class_, class_eval[class_]["TP"], class_eval[class_]["FP"],
         #   class_eval[class_]["FN"]))
 
    # Calculate precision and recall for each class at current threshold              
    for class_ in class_eval:
      tp = class_eval[class_]["TP"]
      fp = class_eval[class_]["FP"]
      fn = class_eval[class_]["FN"]
      if tp == fp == fn == 0:
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

      PR[class_]["precision"].append(precision)   
      PR[class_]["recall"].append(recall)

  #print(PR)

def postprocess(self, net_out, im, save = True):
	"""
	Takes net output, draw net_out, save to disk
	"""
	boxes = self.findboxes(net_out)

	# meta
	meta = self.meta
	threshold = meta['thresh']
	colors = meta['colors']
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
		thick = int((h + w) // 1000)
		if self.FLAGS.json:
			resultsForJSON.append({
				"label": mess, 
				"confidence": float('%.2f' % confidence), 
				"topleft": {"x": left, "y": top}, 
				"bottomright": {"x": right, "y": bot}})
			continue

		cv2.rectangle(imgcv,
			(left, top), (right, bot),
			colors[max_indx], thick)
		cv2.putText(imgcv, mess, (left, top - 12),
			0, 1e-3 * h, colors[max_indx], thick)
	if not save: return imgcv

	outfolder = os.path.join(self.FLAGS.imgdir, 'out')
	img_name = os.path.join(outfolder, os.path.basename(im))
	if self.FLAGS.json:
		textJSON = json.dumps(resultsForJSON)
		textFile = os.path.splitext(img_name)[0] + ".json"
		with open(textFile, 'w') as f:
			f.write(textJSON)
		return

	cv2.imwrite(img_name, imgcv)
