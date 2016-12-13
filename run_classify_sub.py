#!/usr/bin/env python

import mvpa2, sklearn, numpy, os, nibabel, sys
from mvpa2 import *
from mvpa2.datasets.sources import *
from mvpa2.datasets.eventrelated import *
from mvpa2.base.dataset import *
from mvpa2.mappers.detrend import *
from mvpa2.mappers.zscore import *

#where the data live
data_path="/data/akram/MDMRT_scan/"

#Can use this handy openfmri dataset tool to define your dataset
dhandle = OpenFMRIDataset(data_path)

#You can check all participants in this way
#dhandle.get_subj_ids()

#You can check the task descriptions from condition_key.txt
#dhandle.get_task_descriptions()

#You can check all experiment design and model setup info
model = 10
subj = 2

task=1
run_datasets = []

for run_id in dhandle.get_task_bold_run_ids(task)[subj]:
    mask_fname = os.path.join(data_path,'sub'+'{:03d}'.format(subj), 'BOLD', 'task00'+str(task)+'_run001', 'bold_mcf_brain_mask.nii.gz')
    # load design info for this run
    run_events = dhandle.get_bold_run_model(model, subj, run_id)
    #you end up with both task1 and task2 eventns, so subset out just the task 1 event
    run_events=[item for item in run_events if item["task"] == task]
    # load BOLD data for this run (with masking); add 0-based chunk ID
    run_ds = dhandle.get_bold_run_dataset(subj, task, run_id,chunks=run_id -1,mask=mask_fname)
    # convert event info into a sample attribute and assign as 'targets'
    if run_events[len(run_events)-1]['onset']>max(run_ds.sa.time_coords):
        run_events[len(run_events)-1]['onset']=max(run_ds.sa.time_coords)-.00001
    run_ds.sa['targets'] = events2sample_attr(run_events, run_ds.sa.time_coords, noinfolabel='rest')
    run_datasets.append(run_ds)

t1fds = vstack(run_datasets, a=0)

task=2
run_datasets = []

for run_id in dhandle.get_task_bold_run_ids(task)[subj]:
    mask_fname = os.path.join(data_path,'sub002', 'BOLD', 'task00'+str(task)+'_run001', 'bold_mcf_brain_mask.nii.gz')
    # load design info for this run
    run_events = dhandle.get_bold_run_model(model, subj, run_id)
    #you end up with both task1 and task2 eventns, so subset out just the task 1 events
    run_events=[item for item in run_events if item['task'] == task]
    # load BOLD data for this run (with masking); add 0-based chunk ID
    run_ds = dhandle.get_bold_run_dataset(subj, task, run_id,chunks=run_id -1,mask=mask_fname)
    if run_events[len(run_events)-1]['onset']>max(run_ds.sa.time_coords):
        run_events[len(run_events)-1]['onset']=max(run_ds.sa.time_coords)-.00001
    # convert event info into a sample attribute and assign as 'targets'
    run_ds.sa['targets'] = events2sample_attr(run_events, run_ds.sa.time_coords, noinfolabel='rest')
    run_datasets.append(run_ds)

t2fds = vstack(run_datasets, a=0)

#detrender
detrender = PolyDetrendMapper(polyord=1, chunks_attr='chunks')
detrended_t1fds = t1fds.get_mapped(detrender)
detrender = PolyDetrendMapper(polyord=1, chunks_attr='chunks')
detrended_t2fds = t2fds.get_mapped(detrender)

#zscore
zscore(detrended_t1fds, param_est=('targets', ['rest']))
zscore(detrended_t2fds, param_est=('targets', ['rest']))
t1fds = detrended_t1fds
t2fds = detrended_t2fds

#remove rest periods
t1fds = t1fds[t1fds.sa.targets != 'rest']
t2fds = t2fds[t2fds.sa.targets != 'rest']

###############
for ev in events:
        onset = ev['onset'] + onset_shift
        # first sample ending after stimulus onset
        onset_samp_idx = np.argwhere(time_coords[1:] > onset)[0,0]
        # deselect all volume starting before the end of the stimulation
        duration_mask = time_coords < (onset + ev['duration'])
        duration_mask[:onset_samp_idx] = False

run = 1
events = dhandle.get_bold_run_model(model, subj, run)

#you end up with both task1 and task2 eventns, so subset out just the task 1 events
t1events=[item for item in events if item["task"] == 1]

#You can check your events (there should be 70 trials)
for ev in t1events[:70]:
    print ev

#Paths for the nifti images    
bold_fname = os.path.join(data_path,'sub002', 'BOLD', 'task001_run001', 'bold_mcf_brain.nii.gz')
mask_fname = os.path.join(data_path,'sub002', 'BOLD', 'task001_run001', 'bold_mcf_brain_mask.nii.gz')

#load bold data
fds = fmri_dataset(samples=bold_fname,mask=mask_fname)

targets = events2sample_attr(t1events, fds.sa.time_coords,noinfolabel='rest', onset_shift=0.0)

fds = fmri_dataset(samples=bold_fname,mask=mask_fname,chunks=1,targets=targets)
