import csv
from pathlib import Path
REQUIRED={'path','identity'}
def iter_manifest(csv_path,root='.',chunk_size=256):
    if chunk_size<=0: raise ValueError('chunk_size must be positive')
    with Path(csv_path).open(newline='',encoding='utf-8-sig') as f:
        reader=csv.DictReader(f); missing=REQUIRED-set(reader.fieldnames or [])
        if missing: raise ValueError(f'Manifest missing columns: {sorted(missing)}')
        batch=[]
        for n,row in enumerate(reader,2):
            row={k:(v or '').strip() for k,v in row.items()}
            if not row['path'] or not row['identity']: raise ValueError(f'Empty path/identity at CSV row {n}')
            row['absolute_path']=str(Path(root)/row['path']); batch.append(row)
            if len(batch)==chunk_size: yield batch; batch=[]
        if batch: yield batch
