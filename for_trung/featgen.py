##############################################################################
#
# File:         featgen.py
# Date:         Tue 20 Dec 2016  11:46
# Author:       Ken Basye
# Description:  Use Sphinx libraries to generate feature files
#
##############################################################################

import mfcc
import wave
import array
import numpy as np

wavdir = "/Users/kbasye1/Desktop/CS201_F16/code/audio/asr_data/wav"
featdir = "/Users/kbasye1/Desktop/CS201_F16/code/audio/asr_data/kjbfeats"

wav_list = [
"air_000","air_002","air_004","air_006","air_008","air_010","air_012","air_014","air_016","air_018",
"art_000","art_002","art_004","art_006","art_008","art_010","art_012","art_014","art_016","art_018",
"bdl_000","bdl_002","bdl_004","bdl_006","bdl_008","bdl_010","bdl_012","bdl_014","bdl_016","bdl_018",
"dah_000","dah_002","dah_004","dah_006","dah_008","dah_010","dah_012","dah_014","dah_016","dah_018",
"iah_000","iah_002","iah_004","iah_006","iah_008","iah_010","iah_012","iah_014","iah_016","iah_018",
"lim_000","lim_002","lim_004","lim_006","lim_008","lim_010","lim_012","lim_014","lim_016","lim_018"]

def process_one_file(wav_base_name):
    wavefilename = wavdir + "/" + wav_base_name + ".wav"
    print wavefilename
    fh = wave.open(wavefilename, "r")
    sampwidth = fh.getsampwidth()
    print fh.getparams()
    nsamples = fh.getnframes()
    bytes = fh.readframes(nsamples)
    code = "h"
    samples = array.array(code, bytes)
    samples = np.array(samples)
    fh.close()
    print len(samples)
    print samples[12000:12020]
    mfcc_processor = mfcc.MFCC()
    feats = mfcc_processor.sig2s2mfc(samples)
    print len(feats)
    print feats[10:12]
    featfilename = featdir + "/" + wav_base_name + ".feat"
    feats.dump(featfilename)
    checkfeats = np.load(featfilename)
    assert (feats == checkfeats).all()

for wav in wav_list:
    process_one_file(wav)




