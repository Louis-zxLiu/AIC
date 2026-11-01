"""Validate paired input stems and normalized YOLO labels without modifying data."""
from pathlib import Path
import argparse,math,json
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--training',action='store_true');a=p.parse_args()
 modes={}
 for m in ['visible','infrared','depth']:
  paths=[q for q in (a.root/m).iterdir() if q.suffix.lower() in ['.png','.jpg','.jpeg']]
  stems=[q.stem for q in paths];assert len(stems)==len(set(stems)),f'Duplicate stem in {m}';modes[m]=set(stems)
 assert modes['visible'] and all(x==modes['visible'] for x in modes.values()),'Modality stem sets differ'
 boxes=0
 if a.training:
  for stem in sorted(modes['visible']):
   f=a.root/'labels'/(stem+'.txt');assert f.is_file(),str(f)
   for line in f.read_text(encoding='utf-8-sig').splitlines():
    if not line.strip():continue
    v=list(map(float,line.split()));assert len(v)==5 and v[0]==int(v[0]) and 0<=v[0]<12
    assert all(math.isfinite(x) and 0<=x<=1 for x in v[1:]) and v[3]>0 and v[4]>0;boxes+=1
 print(json.dumps({'passed':True,'samples':len(modes['visible']),'training':a.training,'boxes':boxes}))
if __name__=='__main__':main()
