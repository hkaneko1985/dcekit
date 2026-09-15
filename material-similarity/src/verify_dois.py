#!/usr/bin/env python3
"""Read-only registry audit, preserving source spellings and lookup failures."""
import json,csv,urllib.request,urllib.parse,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def check(doi):
 canonical=urllib.parse.unquote(doi)
 url='https://api.crossref.org/works/'+urllib.parse.quote(canonical,safe='')
 base={'doi':doi,'canonical_doi':canonical,'checked_on':'2026-09-14'}
 try:
  with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'material-similarity/0.2 (academic metadata audit)'}),timeout=15) as r:d=json.load(r)['message']
  return {**base,'status':'crossref_resolved' if urllib.parse.unquote(d.get('DOI','')).casefold()==canonical.casefold() else 'metadata_mismatch','title':'; '.join(d.get('title',[])),'URL':d.get('URL','')}
 except Exception as e:return {**base,'status':'lookup_unresolved','title':'','URL':url,'error':str(e)[:150]}
def main():
 path=ROOT/'results/revision/doi_resolution.json'
 old={x['doi']:x for x in json.loads(path.read_text())} if path.exists() else {}
 rows=list(csv.DictReader((ROOT/'results/revision/source_identifiers.csv').open()))
 dois=sorted({r['doi'] for r in rows if r['identifier_status']=='doi_formatted_unverified'})
 out=[]
 for doi in dois:
  if doi in old and old[doi]['status']=='crossref_resolved':out.append(old[doi]);continue
  result=check(doi);out.append(result)
  path.write_text(json.dumps(out+[old[d] for d in dois if d not in {x['doi'] for x in out} and d in old],indent=2)+'\n')
  time.sleep(.5)
 path.write_text(json.dumps(out,indent=2)+'\n')
 from collections import Counter
 print(Counter(x['status'] for x in out),flush=True)
if __name__=='__main__':main()
