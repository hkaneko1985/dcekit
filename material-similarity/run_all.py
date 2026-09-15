#!/usr/bin/env python3
"""Run current source analyses or explicitly regenerate figures from saved tables."""
from pathlib import Path
import argparse,subprocess,sys
ROOT=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tables-only',action='store_true',help='Run available checks and plot references; no fresh data analysis');a=p.parse_args()
 if not a.tables_only and not (ROOT/'data/upstream/nanomine_phase3_records.jsonl.gz').exists():
  raise SystemExit('The public package omits NanoMine API data. Restore a matching authorized snapshot with src/restore_external_data.py, or use --tables-only for saved figures/checks. See data/external_data_manifest.json.')
 commands=[['-m','unittest','discover','-s','tests','-v']]
 if not a.tables_only:commands += [['src/run_validation.py'],['src/check_reproduction.py','--report','outputs/reproduction_check.json'],['src/run_revision_evaluation.py'],['src/run_interval_utility.py'],['src/run_baseline_documentation.py'],['src/check_revision_reproduction.py']]
 resultdir='results' if a.tables_only else 'outputs/reproduced';revisiondir='results/revision' if a.tables_only else 'outputs/revision'
 commands += [['src/make_figures.py','--results-dir',resultdir],['src/make_revision_figures.py','--results-dir',revisiondir]]
 for i,args in enumerate(commands,1):
  print(f'[{i}/{len(commands)}] '+ ' '.join(args),flush=True);subprocess.run([sys.executable,*args],cwd=ROOT,check=True)
 print('Completed '+('saved-table checks/figures only; no data analysis rerun.' if a.tables_only else 'current common/revision analyses and reproduction comparisons.'),flush=True)
if __name__=='__main__':main()
