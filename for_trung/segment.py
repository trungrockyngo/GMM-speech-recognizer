##############################################################################
#
# File:         segment.py
# Date:         Tue 20 Dec 2016  13:41
# Author:       Ken Basye
# Description:  Segment multiple feature files into arrays of features corresponding to a single phoneme
#
##############################################################################

import numpy as np
import os

"""
Every line in the segfile format looks like this:

filename phoneme start_frame end_frame

"""

def process_segfile(segfilename, feat_dir, out_dir):
    with open(segfilename) as infile:
        worklist = dict()
        for line in infile:
            filename, phoneme, start, end = line.split()
            start = int(start)
            end = int(end)
            if filename not in worklist:
                worklist[filename] = list()
            worklist[filename].append((phoneme, start, end))
    for fname, work in worklist.items():
        framedict = dict()
        for (phoneme, start, end) in work:
            feats = np.load(os.path.join(feat_dir, fname))
            workfeats = np.array(feats[start:end])
            if phoneme not in framedict:
                framedict[phoneme] = workfeats
            else:
                framedict[phoneme] = np.concatenate((framedict[phoneme], workfeats), axis=0)
        for phoneme, feat_array in framedict.items():
            outfilename = os.path.join(out_dir, os.path.splitext(fname)[0] + "_" + phoneme + ".feat")
            print("Dumping a total of ", len(feat_array), " features to ", outfilename)
            feat_array.dump(os.path.join(outfilename))
                
                      


    
process_segfile("../kjbfeats/file1.seg", "../kjbfeats", ".")
