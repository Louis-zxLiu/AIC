"""Verify external weights and prediction ZIPs without loading Torch or a GPU."""
from pathlib import Path
import argparse,hashlib,json,math,zipfile
ROOT=Path(__file__).resolve().parents[1]
def sha256(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8<<20),b''):h.update(block)
 return h.hexdigest()
def inspect_submission(path):
 content_hash=hashlib.sha256();boxes=0;empty=0;minimum=1.
 with zipfile.ZipFile(path) as z:
  assert z.testzip() is None,'ZIP CRC check failed'
  names=z.namelist();assert len(names)==len(set(names)),'Duplicate ZIP entries'
  for name in sorted(names):
   assert name==Path(name).name and '/' not in name and '\\' not in name and name.endswith('.txt'),'Expected root-level TXT entries'
   raw=z.read(name);content_hash.update(name.encode('utf-8')+b'\0');content_hash.update(raw);content_hash.update(b'\0')
   lines=[s for s in raw.decode('utf-8').splitlines() if s.strip()]
   assert len(lines)<=100,'More than100 boxes/image';boxes+=len(lines);empty+=not lines
   for line in lines:
    fields=line.split();assert len(fields)==6,'Expected class,cx,cy,w,h,score'
    c=int(fields[0]);values=list(map(float,fields[1:]));assert 0<=c<12
    assert all(math.isfinite(x) and 0<=x<=1 for x in values)
    assert values[2]>0 and values[3]>0 and values[4]>=.25
    minimum=min(minimum,values[4])
 return {'images':len(names),'boxes':boxes,'empty_images':empty,'minimum_score':minimum if boxes else None,'prediction_contents_sha256':content_hash.hexdigest(),'archive_sha256':sha256(path)}
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--checkpoint',type=Path);ap.add_argument('--submission',type=Path);ap.add_argument('--exact',action='store_true',help='Require byte-identical TXT contents to the archived56.132 result')
 a=ap.parse_args();metadata=json.loads((ROOT/'docs/provenance.json').read_text(encoding='utf-8'));reference=json.loads((ROOT/'docs/reference_results.json').read_text())
 result={'metadata_valid':metadata['checkpoint']['epoch_index']==47 and metadata['architecture']['decoder_layers']==4};assert result['metadata_valid']
 if a.checkpoint:
  digest=sha256(a.checkpoint);assert digest==metadata['checkpoint']['sha256'],'Checkpoint differs from archived epoch47 EMA';result['checkpoint_sha256']=digest
 if a.submission:
  item=inspect_submission(a.submission);item['matches_archived_TXT_contents']=item['prediction_contents_sha256']==reference['prediction_contents_sha256']
  if a.exact:assert item['matches_archived_TXT_contents'],'TXT contents differ from the archived result'
  result['submission']=item
 elif a.exact:ap.error('--exact requires --submission')
 print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
