"""Synchronized five-view RGB/IR/Depth preprocessing from the scored run."""
from pathlib import Path
from functools import lru_cache
import numpy as np
import torch
from torch.utils.data import Dataset
from aic.data.modalities import load_visible,load_infrared,load_depth_raw,preprocess_depth
from aic.data.transforms import SyncMultiModalTransform
from .g6 import grid_windows


@lru_cache(maxsize=4)
def raw_triplet(rgb,ir,depth):
 visible=load_visible(rgb);infrared=load_infrared(ir);raw,kind,agreement=load_depth_raw(depth)
 norm,valid,quality=preprocess_depth(raw,kind,agreement)
 return visible,infrared,norm,valid,quality,kind

def collate(batch):
 a,b,c,t=zip(*batch);return torch.stack(a),torch.stack(b),torch.stack(c),list(t)

class RawViews(Dataset):
 def __init__(self,data,stems,grid,overlap,flipped=False):
  self.stems=stems;self.grid=grid;self.overlap=overlap;self.flipped=flipped;self.n=1+grid*grid
  self.paths={m:{p.stem:str(p) for p in (data/m).iterdir() if p.suffix.lower() in ['.png','.jpg','.jpeg']} for m in ['visible','infrared','depth']}
  self.tf=SyncMultiModalTransform(size=[880,1600],train=False)
  for mode in self.paths:assert set(stems)<=self.paths[mode].keys()
 def __len__(self):return len(self.stems)*self.n
 def __getitem__(self,index):
  stem=self.stems[index//self.n];view=index%self.n
  rgb,ir,dep,valid,quality,kind=raw_triplet(*[self.paths[m][stem] for m in ['visible','infrared','depth']])
  h,w=rgb.shape[:2];window=grid_windows(w,h,self.grid,self.overlap)[view];left,top,right,bottom=window
  arrays=[x[top:bottom,left:right] for x in [rgb,ir,dep,valid,quality]]
  if self.flipped:arrays=[np.ascontiguousarray(x[:,::-1]) for x in arrays]
  a,b,c,t=self.tf(*arrays,kind,np.empty((0,4),np.float32),np.empty(0,np.int64))
  t.update(stem=stem,view=view,window=list(window),full_size=[w,h]);return a,b,c,t
