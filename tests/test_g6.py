"""CPU regression coverage for tile boundaries, crowded matching and output caps."""
from pathlib import Path
import sys,unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from aic.best56132.g6 import grid_windows,make_candidates,merge,prediction

def view(window,boxes,scores,labels):
 return {'window':list(window),'boxes':np.asarray(boxes,dtype=float).reshape(-1,4),'scores':scores,'labels':labels}
def candidate(box,score=.8,cls=0,tile=False):return {'box':np.asarray(box,float),'score':score,'class':cls,'tile':tile}

class G6Tests(unittest.TestCase):
 def test_original_640_windows(self):
  self.assertEqual(grid_windows(640,360),[(0,0,640,360),(0,0,356,200),(284,0,640,200),(0,160,356,360),(284,160,640,360)])
 def test_windows_stay_in_image(self):
  for w,h in [(1,1),(1280,720),(1920,1080)]:
   windows=grid_windows(w,h);self.assertEqual(len(windows),5)
   self.assertTrue(all(0<=x1<x2<=w and 0<=y1<y2<=h for x1,y1,x2,y2 in windows))
 def test_internal_tile_edge_is_filtered(self):
  rec={'size':[100,80],'views':[view([40,0,100,80],[[0,10,20,30],[10,10,20,30]],[.9,.8],[0,0])]}
  vv,removed=make_candidates(rec);self.assertEqual(removed,1);np.testing.assert_array_equal(vv[0][0]['box'],[50,10,60,30])
 def test_external_image_edge_is_retained(self):
  rec={'size':[100,80],'views':[view([0,0,100,80],[[0,0,20,30]],[.25],[0])]}
  vv,n=make_candidates(rec);self.assertEqual(n,0);self.assertEqual(len(vv[0]),1)
 def test_invalid_and_subthreshold_candidates_do_not_enter_merge(self):
  rec={'size':[100,80],'views':[view([0,0,100,80],[[1,1,2,2],[1,1,1,2],[1,1,np.nan,2]],[.24999,.9,.9],[0,0,0])]}
  self.assertEqual(make_candidates(rec)[0],[[]])
 def test_two_crowded_people_remain_two(self):
  a=[candidate([10,10,30,30],.8),candidate([12,10,32,30],.7)]
  b=[candidate([10,10,30,30],.9,tile=True),candidate([12,10,32,30],.85,tile=True)]
  result=merge([a,b]);self.assertEqual(len(result),2);self.assertTrue(all(len(c['views'])==2 for c in result))
 def test_different_classes_are_not_merged(self):
  result=merge([[candidate([10,10,30,30],cls=0)],[candidate([10,10,30,30],cls=1)]])
  self.assertEqual(len(result),2)
 def test_full_view_supported_cluster_is_not_demoted(self):
  rec={'size':[100,80],'views':[view([0,0,100,80],[[10,10,30,30]],[.8],[0]),view([0,0,60,50],[[10,10,30,30]],[.6],[0])]}
  rows=prediction(rec);self.assertEqual(len(rows),1);self.assertEqual(rows[0][-1],.8)
 def test_tile_only_cluster_is_demoted(self):
  rec={'size':[100,80],'views':[view([0,0,100,80],[],[],[]),view([0,0,60,50],[[10,10,30,30]],[.65],[0])]}
  rows=prediction(rec);self.assertEqual(rows[0][-1],.45)
 def test_no_predictions_produces_empty_rows(self):
  rec={'size':[100,80],'views':[view([0,0,100,80],[],[],[])]};self.assertEqual(prediction(rec),[])
 def test_max100_and_five_decimal_serialization(self):
  boxes=[[i*5+1,10,i*5+3,30] for i in range(120)]
  rec={'size':[1000,100],'views':[view([0,0,1000,100],boxes,[.6789123]*120,[0]*120)]}
  rows=prediction(rec);self.assertEqual(len(rows),100);self.assertEqual(rows[0][-1],.67891)
  self.assertLess(rows[0][1],rows[-1][1])
if __name__=='__main__':unittest.main()
