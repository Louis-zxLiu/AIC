"""One numerical prediction path shared by validation and submission."""
import numpy as np


def submission_rows(boxes, scores, labels, width, height, score_thresh=.05, max_boxes=100):
    boxes=np.asarray(boxes,dtype=np.float64).reshape(-1,4)
    scores=np.asarray(scores,dtype=np.float64)
    labels=np.asarray(labels,dtype=np.int64)
    if not np.isfinite(boxes).all() or not np.isfinite(scores).all():
        raise ValueError('non-finite detector output')
    keep=np.flatnonzero(scores >= score_thresh)
    keep=keep[np.argsort(-scores[keep],kind='stable')[:max_boxes]]
    boxes,scores,labels=boxes[keep],scores[keep],labels[keep]
    cx=(boxes[:,0]+boxes[:,2])/2/width; cy=(boxes[:,1]+boxes[:,3])/2/height
    bw=(boxes[:,2]-boxes[:,0])/width; bh=(boxes[:,3]-boxes[:,1])/height
    coords=np.round(np.clip(np.stack([cx,cy,bw,bh],1),0,1),5)
    scores=np.round(scores,5)
    good=(coords[:,2]>0)&(coords[:,3]>0)
    return [(int(c),*xywh.tolist(),float(s)) for c,xywh,s in zip(labels[good],coords[good],scores[good])]


def evaluator_rows(rows):
    return [(c,np.array([x-w/2,y-h/2,x+w/2,y+h/2]),s) for c,x,y,w,h,s in rows]
