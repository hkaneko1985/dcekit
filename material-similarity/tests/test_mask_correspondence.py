"""Tests for the re-review's semantic masking error, including repeated steps."""
import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from run_revision_evaluation import setting_fields, mask
from material_similarity import DocumentedValue as V, MaterialInstance as M, ProcessStep as P, ReportingStatus as S, MaterialSimilarityEngine as E
from material_similarity.distance import step_setting_locations

def record(name, types):
    return M(name, process_steps=tuple(P(t, {'Time':V.reported(i+1, kind='numeric', unit_key='duration|min')},
                                        source_id=f'{name}:{i}', setting_provenance={'Time':[{'source':'test'}]})
                                  for i,t in enumerate(types)), process_sequence_status=S.REPORTED)

class CanonicalMaskTests(unittest.TestCase):
    def test_keys_identical_to_distance_flattening(self):
        r=record('a', ['HEAT', 'Cool', ' heat '])
        self.assertEqual(set(setting_fields(r)), set(E([r])._flatten_settings(r)[0]))
        self.assertEqual(list(step_setting_locations(r)), ['heat#0:time','cool#0:time','heat#1:time'])
    def test_interleaving_does_not_mask_unrelated_step(self):
        a=record('a',['mixing','mixing','molding'])
        b=record('b',['mixing','heating','mixing','drying'])
        keys=set(setting_fields(a)) & set(setting_fields(b))
        self.assertEqual(keys, {'mixing#0:time','mixing#1:time'})
        ma,mb=mask(a,keys),mask(b,keys)
        self.assertEqual(set(setting_fields(a))-set(setting_fields(ma)),keys)
        self.assertEqual(set(setting_fields(b))-set(setting_fields(mb)),keys)
        self.assertEqual(mb.process_steps[1].settings['Time'].status,S.REPORTED)
        self.assertEqual(ma.process_steps[2].settings['Time'].status,S.REPORTED)
    def test_missing_side_is_not_in_bilateral_intersection(self):
        a=record('a',['heat']); b=M('b',process_steps=(P('heat',{'Time':V.unknown()}),),process_sequence_status=S.REPORTED)
        self.assertEqual(set(setting_fields(a)) & set(setting_fields(b)),set())
        self.assertEqual(mask(a,set()).process_steps,a.process_steps)
    def test_mask_preserves_step_order_repeats_and_provenance(self):
        a=record('a',['heat','cool','heat']);ma=mask(a,{'heat#1:time'})
        self.assertEqual([s.step_type for s in a.process_steps],[s.step_type for s in ma.process_steps])
        self.assertEqual(ma.process_steps[2].source_id,a.process_steps[2].source_id)
        self.assertEqual(ma.process_steps[2].setting_provenance,a.process_steps[2].setting_provenance)
        v=ma.process_steps[2].settings['Time']
        self.assertEqual((v.kind,v.unit_key,v.value),('numeric','duration|min',None))
        self.assertEqual(ma.process_steps[0],a.process_steps[0])

if __name__=='__main__':unittest.main()
