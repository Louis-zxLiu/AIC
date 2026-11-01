from pathlib import Path
import importlib.util,json,tempfile,unittest,zipfile
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('artifact_check',ROOT/'tools/check_artifacts.py');M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
class ArtifactTests(unittest.TestCase):
 def test_empty_txt_and_valid_row(self):
  with tempfile.TemporaryDirectory() as temp:
   path=Path(temp)/'result.zip'
   with zipfile.ZipFile(path,'w') as z:z.writestr('a.txt','');z.writestr('b.txt','10 0.5 0.5 0.1 0.1 0.25\n')
   r=M.inspect_submission(path);self.assertEqual((r['images'],r['boxes'],r['empty_images']),(2,1,1))
 def test_nested_paths_are_rejected(self):
  with tempfile.TemporaryDirectory() as temp:
   path=Path(temp)/'result.zip'
   with zipfile.ZipFile(path,'w') as z:z.writestr('nested/a.txt','')
   with self.assertRaises(AssertionError):M.inspect_submission(path)
 def test_non_finite_confidence_is_rejected(self):
  with tempfile.TemporaryDirectory() as temp:
   path=Path(temp)/'result.zip'
   with zipfile.ZipFile(path,'w') as z:z.writestr('a.txt','0 .5 .5 .1 .1 nan\n')
   with self.assertRaises(AssertionError):M.inspect_submission(path)
 def test_content_hash_ignores_compression(self):
  with tempfile.TemporaryDirectory() as temp:
   paths=[Path(temp)/f'{i}.zip' for i in range(2)]
   for p,compression in zip(paths,[zipfile.ZIP_STORED,zipfile.ZIP_DEFLATED]):
    with zipfile.ZipFile(p,'w',compression) as z:z.writestr('a.txt','0 .5 .5 .1 .1 .8\n')
   self.assertEqual(M.inspect_submission(paths[0])['prediction_contents_sha256'],M.inspect_submission(paths[1])['prediction_contents_sha256'])
 def test_selected_checkpoint_metadata(self):
  d=json.loads((ROOT/'docs/provenance.json').read_text(encoding='utf-8'));self.assertEqual(d['checkpoint']['epoch_index'],47);self.assertEqual(d['checkpoint']['ema_tensors'],1110)
if __name__=='__main__':unittest.main()
