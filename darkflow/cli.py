from .defaults import argHandler #Import the default arguments
import os
from .net.build import TFNet

import time

def cliHandler(args):
    FLAGS = argHandler()
    FLAGS.setDefaults()
    FLAGS.parseArgs(args)

    # make sure all necessary dirs exist
    def _get_dir(dirs):
        for d in dirs:
            this = os.path.abspath(os.path.join(os.path.curdir, d))
            if not os.path.exists(this): os.makedirs(this)
    _get_dir([FLAGS.imgdir, FLAGS.binary, FLAGS.backup, 
             os.path.join(FLAGS.imgdir,'out'), FLAGS.summary])

    # fix FLAGS.load to appropriate type
    try: FLAGS.load = int(FLAGS.load)
    except: pass

    tfnet = TFNet(FLAGS)
    
    if FLAGS.demo:
        tfnet.camera()
        exit('Demo stopped, exit.')

    if FLAGS.train:
        print('Enter training ...'); tfnet.train()
        if not FLAGS.savepb: 
            exit('Training finished, exit.')

    if FLAGS.savepb:
        print('Rebuild a constant version ...')
        tfnet.savepb(); exit('Done')


    if FLAGS.classify:
        start = time.time()
        tfnet.classify(FLAGS.classify)
        end = time.time() - start
        print("Classification time: {0:.2f}s".format(end))
        exit('Classification finished, exit.')

    tfnet.predict(FLAGS.imgdir)

    if FLAGS.json2tif:
        tfnet.save_prediction_stacks()
        exit("TIF stacks saved")

    if FLAGS.clinic:
        tfnet.OCTpredict()
        exit("Predictions saved")

    if FLAGS.evaluate:
        start = time.time()
        tfnet.evaluate()
        end = time.time() - start
        print("Evaluation time: {0:.2f}s".format(end))




