import csv
import pytest
from app.manifest import iter_manifest

def test_chunk_sizes_and_rows(tmp_path):
 p=tmp_path/'m.csv'
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['path','identity']); w.writeheader(); w.writerows([{'path':f'{i}.jpg','identity':'id'} for i in range(5)])
 batches=list(iter_manifest(p,tmp_path,2))
 assert [len(x) for x in batches]==[2,2,1]
 assert batches[0][0]['absolute_path']==str(tmp_path/'0.jpg')
def test_invalid_chunk_size(tmp_path):
 with pytest.raises(ValueError): list(iter_manifest(tmp_path/'none',chunk_size=0))
def test_empty_label_rejected(tmp_path):
 p=tmp_path/'m.csv'; p.write_text('path,identity\na.jpg,\n')
 with pytest.raises(ValueError,match='Empty'): list(iter_manifest(p))
def test_missing_columns(tmp_path):
 p=tmp_path/'m.csv'; p.write_text('filename,name\na.jpg,x\n')
 with pytest.raises(ValueError,match='missing columns'): list(iter_manifest(p))
