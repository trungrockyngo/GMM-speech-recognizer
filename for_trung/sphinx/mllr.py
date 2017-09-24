"""Adapt acoustic models using maximum-likelihood linear regression (MLLR)

This module implements the MLLR algorithm as described in
"""

# Copyright (c) 2006 Carnegie Mellon University
#
# You may copy and modify this freely under the same terms as
# Sphinx-III

__author__ = "David Huggins-Daines <dhuggins@cs.cmu.edu>"
__version__ = "$Revision$"

import numpy
import s3model
import s3gaucnt
import s3lda
import os

def adapt(model, *accumdirs):
    # Read occupation counts from accumdirs
    mcount, vcount, dnom = s3gaucnt.accumdirs(*accumdirs)
        
