#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 18:23:31 2020

@author: lferiani
"""

import shutil
from tqdm import tqdm
from pathlib import Path

src = Path('/Volumes/behavgenom$/Ida/Data/Hydra/SyngentaStrainScreen/RawVideos/')
dst = Path('/Users/lferiani/OneDrive - Imperial College London/Analysis/SyngentaStrainScreen/RawVideos_yamls/')

yaml_fnames = list(src.rglob('metadata.yaml'))

# %%

for fname in tqdm(yaml_fnames):

    # create target file name
    dst_fname = dst / fname.relative_to(src)
    # create folder if does not exist
    dst_fname.parent.mkdir(exist_ok=True, parents=True)
    dst_fname.touch(exist_ok=True)
    shutil.copy(str(fname), str(dst_fname))

