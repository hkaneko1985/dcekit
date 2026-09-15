"""Regression checks for scientifically consequential adapter and distance errors."""
import copy, gzip, json, unittest
from collections import Counter
from pathlib import Path
from material_similarity import DocumentedValue as V, MaterialInstance as M, ProcessStep as P, ReportingStatus as S, MaterialSimilarityEngine as E, ViewSpec
from material_similarity.normalization import normalize_numeric as N, normalize_setting
from material_similarity.adapters import adapt_nanomine
from material_similarity.distance import NumericScale

class NormalizationTests(unittest.TestCase):
    def test_explicit_time_units(self):
        self.assertEqual(N('time','2 hrs')['value'],120)
        self.assertEqual(N('time','2 fortnights')['kind'],'quarantined')
        self.assertEqual(N('time','2')['reason'],'unit_not_reported')
    def test_fraction_aliases(self):
        for key in ['MassFraction','mass fraction']:
            self.assertAlmostEqual(N(key,'10','Percent; %')['value'],.1)
            self.assertAlmostEqual(N(key,'0.1')['value'],.1)
        self.assertNotEqual(N('MassFraction','.1')['unit_group'],N('VolumeFraction','.1')['unit_group'])
    def test_quantity_not_unit_substring(self):
        self.assertEqual(N('screw diameter length ratio','40')['unit_group'],'screw_diameter_length_ratio|1')
        self.assertEqual(N('SpecificSurfaceArea','20','Percent; %')['kind'],'quarantined')
    def test_numeric_chemical_prefix(self):
        self.assertEqual(normalize_setting('additive','1- decanethiol')['kind'],'categorical')
    def test_conflicting_units(self):
        self.assertEqual(N('time','2 hrs','min')['reason'],'conflicting_unit_declarations')
    def test_explicit_angular_units(self):
        import math
        self.assertAlmostEqual(N('rotational frequency','250 Radian per Minute')['value'],250/(2*math.pi))
        self.assertEqual(N('time','2 Week')['value'],20160)
    def test_temperature_conversion(self):
        self.assertAlmostEqual(N('temperature','373.15 K')['value'],100)
        self.assertEqual(N('temperature','-400 C')['kind'],'quarantined')

class SourceRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=Path(__file__).resolve().parents[1]/'data/upstream/nanomine_phase3_records.jsonl.gz'
        if not path.exists():raise unittest.SkipTest('Acquire the NanoMine source snapshot to run source regressions')
        with gzip.open(path,'rt') as f:cls.rows=[json.loads(x) for x in f]
    def test_all_eighteen_duplicate_records_are_lossless_and_order_invariant(self):
        affected=0
        for row in self.rows:
            duplicates=[s for s in row['steps'] if max(Counter(x['key'] for x in s['settings']).values(),default=0)>1]
            if not duplicates:continue
            affected+=1
            with self.subTest(record=row['sample_id']):
                a=adapt_nanomine(row); reversed_row=copy.deepcopy(row)
                for step in reversed_row['steps']:step['settings'].reverse()
                b=adapt_nanomine(reversed_row)
                self.assertEqual(a,b)
                e=E([a]); d=e.compare(a,b,ViewSpec('settings',{'settings':1})).composite
                self.assertEqual(d.reported,0)
                self.assertEqual(sum(len(items) for s in a.process_steps for items in s.setting_provenance.values()),sum(len(s['settings']) for s in row['steps']))
        self.assertEqual(affected,18)
    def test_barrel_four_values_preserved(self):
        r=adapt_nanomine(next(x for x in self.rows if x['sample_id']=='l210-s2-zhang-2014'))
        v=next(s.settings['barrel temperature'] for s in r.process_steps if 'barrel temperature' in s.settings)
        self.assertEqual(v.value,(150.,160.,165.,180.))
    def test_unknown_zone_correspondence_retains_range(self):
        r=adapt_nanomine(next(x for x in self.rows if x['sample_id']=='l210-s2-zhang-2014'))
        d=E([r]).compare(r,r,ViewSpec('settings',{'settings':1})).composite
        self.assertEqual(d.reported,0)
        self.assertGreater(d.upper,d.lower)
    def test_primary_excludes_characterization(self):
        r=adapt_nanomine(self.rows[0]);self.assertFalse(any('density' in k or 'width' in k for k in r.material_identity))

class MissingnessAlignmentTests(unittest.TestCase):
    def test_bilateral_missingness_fixed_universe(self):
        a=M('a',process_steps=(P('heat',{'temperature':V.reported(100,kind='numeric',unit_key='T'),'time':V.reported(0,kind='numeric',unit_key='t')}),),process_sequence_status=S.REPORTED)
        b=M('b',process_steps=(P('heat',{'temperature':V.reported(100,kind='numeric',unit_key='T'),'time':V.reported(10,kind='numeric',unit_key='t')}),),process_sequence_status=S.REPORTED)
        def masked(r):return M(r.record_id,process_steps=(P('heat',{'temperature':r.process_steps[0].settings['temperature'],'time':V.unknown()}),),process_sequence_status=S.REPORTED)
        v=ViewSpec('settings',{'settings':1}); scales={'T':NumericScale(100),'t':NumericScale(10)}
        active=E([a,b],numeric_scales=scales);fixed=E([a,b],numeric_scales=scales,feature_policy='fixed_schema')
        self.assertEqual(active.compare(a,b,v).composite.reported,.5)
        self.assertEqual(active.compare(masked(a),masked(b),v).composite.width,0)
        d=fixed.compare(masked(a),masked(b),v).composite
        self.assertEqual((d.lower,d.upper),(0,.5))
    def test_sequence_alignment_symmetry_with_insertion(self):
        def step(t,x):return P(t,{'time':V.reported(x,kind='numeric',unit_key='t')})
        a=M('a',process_steps=(step('heat',1),step('cool',2),step('heat',3)),process_sequence_status=S.REPORTED)
        b=M('b',process_steps=(step('heat',9),step('heat',1),step('cool',2),step('heat',3)),process_sequence_status=S.REPORTED)
        e=E([a,b],settings_alignment='sequence');v=ViewSpec('s',{'settings':1})
        self.assertEqual(e.compare(a,b,v).composite,e.compare(b,a,v).composite)
        self.assertNotEqual(e.compare(a,b,v).composite.reported,E([a,b]).compare(a,b,v).composite.reported)
    def test_empty_explicit_scales_not_refitted(self):
        r=M('r');self.assertEqual(E([r],numeric_scales={}).numeric_scales,{})

if __name__=='__main__':unittest.main()
