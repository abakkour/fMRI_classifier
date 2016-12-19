#!/usr/bin/env python

import mvpa2, sklearn, numpy, os, nibabel, sys
from mvpa2.suite import *

#where the data live
#data_path="/data/akram/MDMRT_scan/"
data_path="/Volumes/hypatia/akram/MDMRT_scan/"

#wb_mask_fname="/usr/share/fsl/5.0/data/standard/MNI152_T1_2mm_brain_mask.nii.gz"
wb_mask_fname="/usr/local/fsl/data/standard/MNI152_T1_2mm_brain_mask.nii.gz"

sub = "sub002"

#Paths for the nifti images    
beta_t1_fname = os.path.join(data_path,sub, 'model/model100/task001_LSS_std.nii.gz')
beta_t2_fname = os.path.join(data_path,sub, 'model/model100/task002_LSS_std.nii.gz')
mask_fname = os.path.join(data_path,'rois', sub+'_hippo.nii.gz')

#Paths to the attibute files
attr1 = SampleAttributes(os.path.join(data_path,sub,'model/model100/task001_attr.txt'))
attr2 = SampleAttributes(os.path.join(data_path,sub,'model/model100/task002_attr.txt'))

#load datasets
ds1 = fmri_dataset(samples=beta_t1_fname, targets=attr1.targets, chunks=attr1.chunks, mask = mask_fname)
ds2 = fmri_dataset(samples=beta_t2_fname, targets=attr2.targets, chunks=attr2.chunks, mask = mask_fname)

#zscore datasets
zscore(ds1,chunks_attr='chunks',dtype='float32')
zscore(ds2,chunks_attr='chunks',dtype='float32')


#specify classifier
clf = LinearCSVMC()

#set up three-fold cross validation
cvte1 = CrossValidation(clf, NFoldPartitioner(), errorfx=lambda p, t: np.mean(p == t),enable_ca=['stats'])
cvte2 = CrossValidation(clf, NFoldPartitioner(), errorfx=lambda p, t: np.mean(p == t),enable_ca=['stats'])

#run cross validation on task1+2 whole-brain data
t1cv_results = cvte1(ds1)
t2cv_results = cvte2(ds2)


##searchlight

#load datasets
ds1 = fmri_dataset(samples=beta_t1_fname, targets=attr1.targets, chunks=attr1.chunks, mask = wb_mask_fname)
ds2 = fmri_dataset(samples=beta_t2_fname, targets=attr2.targets, chunks=attr2.chunks, mask = wb_mask_fname)

#zscore datasets
zscore(ds1,chunks_attr='chunks',dtype='float32')
zscore(ds2,chunks_attr='chunks',dtype='float32')

cvte = CrossValidation(clf, NFoldPartitioner(), errorfx=lambda p, t: np.mean(p == t),enable_ca=['stats'])

sl = sphere_searchlight(cvte, radius=6, postproc=mean_sample())
t1_sl_results = sl(ds1)
t1_niftiresults = map2nifti(t1_sl_results, imghdr=ds1.a.imghdr)
niftiresults.to_filename(os.path.join(data_path,sub,'model/model100/searchlight_results_2mm_task001_3class.nii.gz'))
