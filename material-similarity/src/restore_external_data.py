#!/usr/bin/env python3
"""Restore authorized matching NanoMine inputs from a local archive; no network."""
import argparse,hashlib,json,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',type=Path,required=True);a=p.parse_args()
 manifest=json.loads((ROOT/'data/external_data_manifest.json').read_text());restored=[];unavailable=[]
 with zipfile.ZipFile(a.archive) as z:
  for row in manifest['files']:
   if row['kind']=='generated_audit':continue
   rel=row['path'];target=(ROOT/rel).resolve()
   if not target.is_relative_to(ROOT):raise ValueError('Unsafe manifest path')
   matches=[n for n in z.namelist() if n==rel or n.endswith('/'+rel)]
   valid=[]
   for name in matches:
    b=z.read(name)
    if hashlib.sha256(b).hexdigest()==row['sha256']:valid.append(b)
   if not valid:
    unavailable.append(rel);continue
   target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(valid[0]);restored.append(rel)
 print(json.dumps({'restored':restored,'not_found_or_hash_mismatch':unavailable},indent=2))
 for row in manifest['files']:
  if not row['required_for_current_analysis']:continue
  path=ROOT/row['path']
  if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:
   raise SystemExit('Required matching snapshot not restored; no live substitute was accepted.')
 print('Required snapshot available. Run python run_all.py. Audited interchange and raw-value audits can be regenerated with src/audit_sources.py.')
if __name__=='__main__':main()
