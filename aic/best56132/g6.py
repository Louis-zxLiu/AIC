"""Frozen G6 postprocessing from the56.132 run, with no Torch dependency."""
import math
import numpy as np
from scipy.optimize import linear_sum_assignment
from aic.eval.map_evaluator import box_iou_matrix
from aic.eval.submission_postprocess import submission_rows


def grid_windows(width,height,grid=2,overlap=.2):
 w=min(width,math.ceil(width/(grid-(grid-1)*overlap)))
 h=min(height,math.ceil(height/(grid-(grid-1)*overlap)))
 xs=np.rint(np.linspace(0,width-w,grid)).astype(int);ys=np.rint(np.linspace(0,height-h,grid)).astype(int)
 return [(0,0,width,height)]+[(int(x),int(y),int(x+w),int(y+h)) for y in ys for x in xs]

def make_candidates(record):
 w,h=record['size'];result=[];removed=0
 for r in record['views']:
  left,top,right,bottom=r['window'];candidates=[]
  for box,score,cls in zip(r['boxes'],r['scores'],r['labels']):
   if score<.25 or not np.isfinite(box).all() or box[2]<=box[0] or box[3]<=box[1]:continue
   clipped=(left>0 and box[0]<=1.5) or (top>0 and box[1]<=1.5) or (right<w and box[2]>=right-left-1.5) or (bottom<h and box[3]>=bottom-top-1.5)
   if clipped:removed+=1;continue
   candidates.append({'class':int(cls),'box':box.astype(float)+np.asarray([left,top,left,top]),'score':float(score),
                      'tile':r['window']!=[0,0,w,h]})
  result.append(candidates)
 return result,removed

def routing_score(c,w,h):
 b=c['box'];side=math.sqrt(max(0,(b[2]-b[0])*1600/w*(b[3]-b[1])*880/h))
 large=min(1.,max(0.,(side-32)/64))
 return c['score']*((1.-.25*large) if c['tile'] else (.75+.25*large))

def merge(views,iou=.6,coordinates='weighted',routing=False,size=None):
 clusters=[]
 for view,candidates in enumerate(views):
  candidates=[{**c,'score':routing_score(c,*size) if routing else c['score']} for c in candidates]
  used=set()
  for cls in range(12):
   ids=[i for i,c in enumerate(candidates) if c['class']==cls]
   current=[i for i,c in enumerate(clusters) if c['class']==cls]
   if not ids or not current:continue
   overlaps=box_iou_matrix(np.asarray([clusters[i]['box'] for i in current]),np.asarray([candidates[i]['box'] for i in ids]))
   valid=overlaps>=iou;rr,cc=linear_sum_assignment(np.where(valid,1000+overlaps,0),maximize=True)
   for r,c in zip(rr,cc):
    if not valid[r,c]:continue
    i=current[r];j=ids[c];a=clusters[i];b=candidates[j];weight=b['score']
    if coordinates=='weighted':a['box']=(a['box']*a['weight']+b['box']*weight)/(a['weight']+weight)
    elif weight>a['score']:a['box']=b['box'].copy()
    a['weight']+=weight;a['score']=max(a['score'],weight);a['views'].append(view);used.add(j)
  for j,b in enumerate(candidates):
   if j not in used:clusters.append({**b,'box':b['box'].copy(),'weight':b['score'],'views':[view]})
 return clusters

def prediction(record):
    views,removed=make_candidates(record)
    clusters=merge(views,iou=.6,coordinates='weighted',routing=False,size=record['size'])
    for c in clusters:
        if 0 not in c['views']:c['score']=.25+(c['score']-.25)*.5
    rows=submission_rows(np.asarray([c['box'] for c in clusters],float).reshape(-1,4),
        np.asarray([c['score'] for c in clusters]),np.asarray([c['class'] for c in clusters],int),*record['size'],.25,100)
    return rows
