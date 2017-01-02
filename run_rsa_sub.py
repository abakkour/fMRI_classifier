#!/usr/bin/env python

import mvpa2, sklearn, os, nibabel, sys
import pylab as pl
import numpy as np
from mvpa2.suite import *

# little helper function to plot dissimilarity matrices
# since we are using correlation-distance, we use colorbar range of [0,2]
def plot_mtx(mtx, labels, title):
    pl.figure()
    pl.imshow(mtx, interpolation='nearest')
    pl.xticks(range(len(mtx)), labels, rotation=-45)
    pl.yticks(range(len(mtx)), labels)
    pl.title(title)
    pl.clim((0, 2))
    pl.colorbar()


#where the data live
data_path="/vega/psych/users/ab4096/MDMRT_scan/"

wb_mask_fname="/vega/psych/app/fsl/data/standard/MNI152_T1_2mm_brain_mask.nii.gz"

sub = str(sys.argv[1])  #"sub002"

#Paths for the nifti images    
beta_t1_fname = os.path.join(data_path,sub, 'model/model101/task001_LSS_std.nii.gz')
beta_t2_fname = os.path.join(data_path,sub, 'model/model101/task002_LSS_std.nii.gz')
mask_fname = os.path.join(data_path,'rois', sub+'_hippo.nii.gz')

#Paths to the attibute files
attr1 = SampleAttributes(os.path.join(data_path,sub,'model/model101/task001_attr.txt'))
attr2 = SampleAttributes(os.path.join(data_path,sub,'model/model101/task002_attr.txt'))

#load datasets
ds1 = fmri_dataset(samples=beta_t1_fname, targets=attr1.targets, chunks=attr1.chunks, mask = mask_fname)
ds2 = fmri_dataset(samples=beta_t2_fname, targets=attr2.targets, chunks=attr2.chunks, mask = mask_fname)

#zscore datasets
zscore(ds1,chunks_attr='chunks',dtype='float32')
zscore(ds2,chunks_attr='chunks',dtype='float32')

# compute a dataset with the mean samples for all conditions
mtgs1 = mean_group_sample(['targets'])
mtds1 = mtgs1(ds1)

mtgs2 = mean_group_sample(['targets'])
mtds2 = mtgs2(ds2)

# basic ROI RSA -- dissimilarity matrix for the entire ROI
dsm = rsa.PDist(square=True)
res1 = dsm(mtds1)
plot_mtx(res1, mtds1.sa.targets, 'ROI pattern correlation distances')

res2 = dsm(mtds2)
plot_mtx(res2, mtds2.sa.targets, 'ROI pattern correlation distances')
