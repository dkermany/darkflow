import os
import time
import numpy as np
import tensorflow as tf
import pickle
import ujson as json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from pprint import pprint
from multiprocessing.pool import ThreadPool
from copy import deepcopy
from glob import glob
from ..utils.box2 import IOU
import line_profiler
import tifffile

train_stats = (
    'Training statistics: \n'
    '\tLearning rate : {}\n'
    '\tBatch size    : {}\n'
    '\tEpoch number  : {}\n'
    '\tBackup every  : {}'
)
pool = ThreadPool()

def _save_ckpt(self, step, loss_profile):
    file = '{}-{}{}'
    model = self.meta['name']

    profile = file.format(model, step, '.profile')
    profile = os.path.join(self.FLAGS.backup, profile)
    with open(profile, 'wb') as profile_ckpt: 
        pickle.dump(loss_profile, profile_ckpt)

    ckpt = file.format(model, step, '')
    ckpt = os.path.join(self.FLAGS.backup, ckpt)
    self.say('Checkpoint at step {}'.format(step))
    self.saver.save(self.sess, ckpt)


def train(self):
    # process "steps" and "scales" for learning rate decay   
    arg_steps = np.array(self.FLAGS.steps[1:-1].split(',')).astype(np.int32)
    arg_scales = np.array(self.FLAGS.scales[1:-1].split(',')).astype(np.float32)
    lr = self.FLAGS.lr
 
    loss_ph = self.framework.placeholders
    loss_mva = None; profile = list()

    batches = self.framework.shuffle()
    loss_op = self.framework.loss

    for i, (x_batch, datum) in enumerate(batches):
        if not i: self.say(train_stats.format(
            self.FLAGS.lr, self.FLAGS.batch,
            self.FLAGS.epoch, self.FLAGS.save
        ))

        feed_dict = {
            loss_ph[key]: datum[key] 
                for key in loss_ph }
        feed_dict[self.inp] = x_batch
        feed_dict.update(self.feed)

        # update learning rate at user-specified steps
        feed_dict[self.learning_rate] = lr
        idx = np.where(arg_steps[:] == i + 1)[0]
        if len(idx):
            new_lr = lr * arg_scales[idx][0]
            lr = new_lr
            feed_dict[self.learning_rate] = lr  
            print("\nSTEP {} - UPDATED LEARNING RATE TO {:.6}".format(i+1, new_lr))

        fetches = [self.train_op, loss_op, self.summary_op] 
        fetched = self.sess.run(fetches, feed_dict)
        loss = fetched[1]

        if loss_mva is None: loss_mva = loss
        loss_mva = .9 * loss_mva + .1 * loss
        step_now = self.FLAGS.load + i + 1

        self.writer.add_summary(fetched[2], step_now)

        form = 'step {} - loss {} - moving ave loss {}'
        self.say(form.format(step_now, loss, loss_mva))
        profile += [(loss, loss_mva)]

        ckpt = (i+1) % (self.FLAGS.save // self.FLAGS.batch)
        args = [step_now, profile]
        if not ckpt: _save_ckpt(self, *args)

    if ckpt: _save_ckpt(self, *args)

def return_predict(self, im):
    assert isinstance(im, np.ndarray), \
        'Image is not a np.ndarray'
    h, w, _ = im.shape
    im = self.framework.resize_input(im)
    this_inp = np.expand_dims(im, 0)
    feed_dict = { self.inp: this_inp }

    out = self.sess.run(self.out, feed_dict)[0]
    boxes = self.framework.findboxes(out)
    threshold = self.FLAGS.threshold
    boxesInfo = list()
    for box in boxes:
        tmpBox = self.framework.process_box(box, h, w, threshold)
        if tmpBox is None:
            continue
        boxesInfo.append({
            "label": tmpBox[4],
            "confidence": tmpBox[6],
            "topleft": {
                "x": tmpBox[0],
                "y": tmpBox[2]},
            "bottomright": {
                "x": tmpBox[1],
                "y": tmpBox[3]}
        })
    return boxesInfo

import math

def predict(self, inp_path, verbose=False):
    all_inps = os.listdir(inp_path)
    all_inps = [i for i in all_inps if self.framework.is_inp(i)]
    if not all_inps:
        msg = 'Failed to find any images in {} .'
        exit('Error: {}'.format(msg.format(inp_path)))

    batch = min(self.FLAGS.batch, len(all_inps))

    # predict in batches
    n_batch = int(math.ceil(len(all_inps) / batch))
    for j in tqdm(range(n_batch)):
        from_idx = j * batch
        to_idx = min(from_idx + batch, len(all_inps))

        # collect images input in the batch
        inp_feed = list(); new_all = list()
        this_batch = all_inps[from_idx:to_idx]
        for inp in this_batch:
            new_all += [inp]
            this_inp = os.path.join(inp_path, inp)
            this_inp = self.framework.preprocess(this_inp)
            expanded = np.expand_dims(this_inp, 0)
            inp_feed.append(expanded)
        this_batch = new_all

        # Feed to the net
        feed_dict = {self.inp : np.concatenate(inp_feed, 0)}
        if verbose: self.say('Forwarding {} inputs ...'.format(len(inp_feed)))    
        start = time.time()
        out = self.sess.run(self.out, feed_dict)
        stop = time.time(); last = stop - start
        if verbose: self.say('Forwarding time = {}s / {} inps = {} ips'.format(
            last, len(inp_feed), len(inp_feed) / last))

        # Post processing
        if verbose: self.say('Post processing {} inputs ...'.format(len(inp_feed)))
        start = time.time()
        threshold = self.FLAGS.threshold
        pool.map(lambda p: (lambda i, prediction:
            self.framework.postprocess(
               prediction, os.path.join(inp_path, this_batch[i]), threshold))(*p),
            enumerate(out))
        stop = time.time(); last = stop - start
        if verbose: self.say('Post processing time = {}s / {} inps = {} ips'.format(
            last, len(inp_feed), len(inp_feed) / last))

        

def evaluate(self, binary=False):
  _labels = self.meta["labels"]
  AP = {}

  # Load gt boxes
  gt_boxes = self.framework.parse()
  gt_boxes = { image[0]: image[1] for image in gt_boxes }
  #gt_labels = set(gt_boxes.keys())
  #print(gt_labels)

  # Load predicted boxes
  outfolder = os.path.join(self.FLAGS.imgdir, "out")
  json_paths = glob("{}/*.json".format(outfolder))
  assert len(json_paths) == len(gt_boxes.keys())

  # Loop through each of the objects in labels.txt
  for current_label in _labels:
    print("\n{}".format(current_label))
    class_eval = { "precision": [], "recall": [], "threshold": [] }

    # Loop through all thresholds
    for threshold in tqdm(np.arange(0.05, 1.00, 0.05)):
      threshold = min(threshold, 1.00)
      tally = { "tp": 0, "fp": 0, "fn": 0 }

      # Loop through predicted box outputs
      for jsfilename in json_paths:
        image_filename = "{}.jpeg".format(os.path.basename(jsfilename).split(".")[0])
        assert image_filename in gt_boxes.keys()#gt_labels
        #print("Image #{}: {}".format(j, image_filename))

        # Extract gt_info
        w, h, current_gt = gt_boxes[image_filename]

        gt_labels = set([ b[0] for b in current_gt ])

        # Create GT Variables for tracking
        gt_check = [ False for b in current_gt if b[0] == current_label ]
        gt_check_list = [ b for b in current_gt if b[0] == current_label ]

        # Load json predictions as a list of dictionaries
        with open(jsfilename) as f:
          js = json.load(f)

        # Loop over each bounding box prediction
        relevant_js = [ bb for bb in js if not bb["confidence"] < threshold and bb["label"] == current_label ]
        for bb in relevant_js:
          #if bb["confidence"] < threshold: continue
          #label = bb["label"]
          # x increases to the right, y increases down
          #xmin, ymin = bb["topleft"]["x"], bb["topleft"]["y"]
          #xmax, ymax = bb["bottomright"]["x"], bb["bottomright"]["y"]

          #if label != current_label: continue

          # if predicted label is not 
          if bb["label"] not in gt_labels:
            tally["fp"] += 1
            continue

          # Create dict to feed into IOU function
          # xmin, xmax, ymin, ymax
          pred_feed = { "xn": bb["topleft"]["x"],
                        "xx": bb["bottomright"]["x"], 
                        "yn": bb["topleft"]["y"], 
                        "yx": bb["bottomright"]["y"] }

          # Loop over each ground-truth box
          for i, box_gt in enumerate(gt_check_list):
            # label, left, top, right, bottom (ymax = bottom)
            #_label, _xmin, _ymin, _xmax, _ymax = box_gt

            # Create dict to feed into IOU function
            gt_feed = { "xn": box_gt[1], "xx": box_gt[3], "yn": box_gt[2], "yx": box_gt[4] }

            try:
              iou = IOU(pred_feed, gt_feed)
            except AssertionError:
              print("IOU Assertion Error")
              #print("pred_feed")
              #pprint(pred_feed)
              #print("gt_feed")
              #pprint(gt_feed)
              #print("\n")
              continue
           
            if iou >= 0.3 and not gt_check[i]:
              tally["tp"] += 1
              gt_check[i] = True
              break
            # if prediction matches ground-truth already discovered, ignore
            elif iou >= 0.4 and gt_check[i]:
              continue

        tally["fn"] += gt_check.count(False)

      #pprint(tally)
      # Calculate precision and recall at each threshold
      tp = tally["tp"]
      fp = tally["fp"]
      fn = tally["fn"]

      if tp == fp == fn == 0:
        # print("Warning: No predictions or ground-truth annotations")
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
      class_eval["threshold"].append(threshold)

    # add end points
    print("\n{}".format(current_label))
    for tup in zip(class_eval["threshold"], class_eval["precision"], class_eval["recall"]):
      print(tup)

    class_eval["precision"].append(1.0)
    class_eval["precision"].insert(0, 0.0)
    class_eval["recall"].append(0.0)
    class_eval["recall"].insert(0, 1.0)
    #pprint(class_eval)
    AP[current_label] = -1.0 * np.trapz(class_eval["precision"], x=class_eval["recall"])

    plt.plot(class_eval["recall"], class_eval["precision"])
  
  plt.legend()
  plt.ylabel("Precision")
  plt.xlabel("Recall")
  plt.title("Precision-Recall Curve")
  plt.savefig("PR.png", bbox_inches="tight")

  pprint(AP)
  mAP = sum(list(AP.values())) / float(len(list(AP.values())))
  print("mAP: {}".format(mAP))

def OCTpredict(self):
  outfolder = os.path.join(self.FLAGS.imgdir, "out")
  assert os.path.exists(outfolder)

  json_paths = glob("{}/*.json".format(outfolder))
  outfolder = os.path.join(outfolder, "best")
  if not os.path.exists(outfolder): os.makedirs(outfolder)

  for jsfilename in tqdm(json_paths):
    basename = os.path.basename(os.path.normpath(jsfilename)).split(".")[0]
    image_filename = "{}/{}.jpeg".format(self.FLAGS.imgdir, basename)
    assert os.path.isfile(image_filename)
    
    with open(jsfilename) as f:
      js = json.load(f)
 
    output = self.framework.postprocess_OCT(js, image_filename)
    savefile = "{}/{}-{}{}".format(outfolder, basename, "prediction", ".jpeg")
    cv2.imwrite(savefile, output)
  print(savefile)
  print(output.shape)

#@profile
def save_prediction_stacks(self):
  outfolder = os.path.join(self.FLAGS.imgdir, "out")
  tif_output_dir = os.path.join(outfolder, "tif")
  if not os.path.exists(tif_output_dir): os.makedirs(tif_output_dir)

  json_paths = glob("{}/*.json".format(outfolder))
  for jsfilename in tqdm(json_paths):
    basename = os.path.basename(os.path.normpath(jsfilename)).split(".")[0]
    image_filename = "{}/{}.jpeg".format(self.FLAGS.imgdir, basename)
    assert os.path.isfile(image_filename)
    
    with open(jsfilename) as f:
      js = json.load(f)
 
    current_stack = self.framework.postprocess_tif(js, image_filename)
    tif_output = "{}/{}.tif".format(tif_output_dir, basename)
    tifffile.imsave(tif_output, current_stack)
    

def classify(self, path):
  gt_classes = {}
  summary = []
  predicted = {}

  # Generate JSON predictions for subdirectories
  # And Create a dict with keys classification labels (e.g. DME, CNV, DRUSEN, NORMAL). Values = list of filenames
  # and # Create dict with classification keys with empty lists to store predictions
  for folder in glob("{}/*/".format(path)):
    class_name = os.path.basename(os.path.normpath(folder))
    print("Generating prediction for {}".format(class_name))
    gt_classes[class_name] = [ os.path.basename(f).split(".")[0] for f in glob("{}/*.jpeg".format(folder)) ]
    predicted[class_name] = []
    self.predict(folder)

  print("Ground-Truth Classes")
  for key, val in gt_classes.items():
    print("{}: {} images".format(key, len(val)))

  ROC = { "sensitivity": [], "specificity": [] }
  for threshold in tqdm(np.arange(0.00, 1.05, 0.05)):
    threshold = min(threshold, 1.00)
    predicted = { key: [] for key, _ in predicted.items() }
    sensitivity = 0.0
    sensitivity_n = 0
    specificity = 0.0
    specificity_n = 0 
    for current_class, filenames in gt_classes.items():
      for filename in filenames:
        json_filename = os.path.join(path, current_class, "out", "{}.json".format(filename))
      
        with open(json_filename) as f:
          js = json.load(f)
        objects = set([ bb["label"] for bb in js if bb["confidence"] >= threshold ])
        if "CNV" in objects or "PED" in objects or "SRH" in objects:
          # Predicted CNV
          predicted["CNV"].append(filename)
        elif "IRF" in objects:
          if "GA" in objects:
            # Predicted CNV
            predicted["CNV"].append(filename)
          else:
            # Predicted DME
            predicted["DME"].append(filename)
        elif "DRU" in objects:
          # Predicted DRUSEN
          predicted["DRUSEN"].append(filename)
        else:
          # Predicted NORMAL
          predicted["NORMAL"].append(filename)

    pred_referrals = predicted["CNV"] + predicted["DME"]
    gt_referrals = gt_classes["CNV"] + gt_classes["DME"]

    pred_normal = predicted["DRUSEN"] + predicted["NORMAL"]
    gt_normal = gt_classes["DRUSEN"] + gt_classes["NORMAL"]

    for filename in gt_referrals:
      sensitivity_n += 1
      if filename in pred_referrals:
        sensitivity += 1

    for filename in gt_normal:
      specificity_n += 1
      if filename in pred_normal:
        specificity += 1

    #summary.append((threshold, sensitivity, specificity, sensitivity_n, specificity_n))
    sens_count = int(sensitivity)
    spec_count = int(specificity)

    if sensitivity_n > 0: sensitivity /= float(sensitivity_n)
    else: sensitivity = 0.0

    if specificity > 0: specificity /= float(specificity_n)
    else: specificity = 0.0

    ROC["sensitivity"].append(sensitivity)
    ROC["specificity"].append(specificity)
    summary.append((threshold, sensitivity, specificity))

  # Pad sensitivity and specificity
  ROC["sensitivity"].append(0.0)
  ROC["sensitivity"].insert(0, 1.0)
  ROC["specificity"].append(1.0)
  ROC["specificity"].insert(0, 0.0) 

  pprint(summary)
  print("Area under ROC: {0:.3f}".format(np.trapz(ROC["sensitivity"], x=ROC["specificity"])))
  plt.plot(reversed(ROC["specificity"]), ROC["sensitivity"])
  plt.legend()
  plt.ylabel("Sensitivity")
  plt.xlabel("1 - Specificity")
  plt.title("Receiver Operating Characteristic Curve")
  plt.savefig("ROC.png", bbox_inches="tight")
 











