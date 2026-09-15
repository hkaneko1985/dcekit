"""Consistency checks, separate from empirical evaluations; no success thresholds."""
import csv,hashlib,json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class ValidationArtifactTests(unittest.TestCase):
 def test_retrospective_scope(self):
  d=json.loads((ROOT/'results/phase4_claims_lock.json').read_text())
  self.assertEqual(d['total_material_instances_across_tracks'],8827)
  self.assertEqual(d['dataset_counts']['NanoMine']['samples'],832)
 def test_api_summary_counts_and_version(self):
  d=json.loads((ROOT/'results/api_validation_summary.json').read_text())
  self.assertEqual(d['api_version'],'0.2.2');self.assertEqual(d['validation_records'],183)
  self.assertEqual(d['material_system_visualization']['selected_records'],72)
  self.assertEqual(d['process_method_visualization']['selected_records'],120)
 def test_htem_csv_summary_agree(self):
  d=json.loads((ROOT/'results/api_validation_summary.json').read_text())
  r=next(x for x in csv.DictReader((ROOT/'results/api_htem_validation.csv').open()) if x['view']=='htem_composition_settings' and x['bound']=='reported')
  self.assertAlmostEqual(float(r['ari_vs_reported_composition']),d['htem_validation']['reported_composite_ari_vs_composition'])
 def test_representative_pair_summary_agree(self):
  d=json.loads((ROOT/'results/api_validation_summary.json').read_text())
  r=list(csv.DictReader((ROOT/'results/representative_pair_views.csv').open()))
  self.assertEqual(len(r),5)
  for x in r:
   self.assertLessEqual(float(x['optimistic']),float(x['pessimistic']))
   self.assertGreaterEqual(float(x['common_reported_fraction']),0)
 def test_audit_manifest_scope(self):
  d=json.loads((ROOT/'results/revision/audit_summary.json').read_text())
  self.assertEqual(d['subsets']['full']['duplicate_setting_records'],18)
  self.assertEqual(d['subsets']['full']['placeholder_records'],53)
  e=json.loads((ROOT/'config/revision_evaluation.json').read_text())
  self.assertEqual(e['heldout_method']['k'],5)
 def test_no_control_characters(self):
  for p in ROOT.rglob('*.md'):
   self.assertFalse([c for c in p.read_text() if ord(c)<32 and c not in '\n\t'],str(p))
 def test_available_frozen_inputs_match_manifest(self):
  d=json.loads((ROOT/'data/upstream_manifest.json').read_text());self.assertEqual(len(d['files']),8)
  excluded=set(json.loads((ROOT/'data/external_data_manifest.json').read_text()).get('excluded_paths',[])) if (ROOT/'data/external_data_manifest.json').exists() else set()
  for row in d['files']:
   p=ROOT/row['path']
   if not p.exists() and row['path'] in excluded:continue
   self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),row['sha256'])
if __name__=='__main__':unittest.main()
