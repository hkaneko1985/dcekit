"""Explicit quantity/unit harmonization for source-reported settings.

No units or missing settings are inferred from composition or correlations.
Unrecognized or inconsistent numeric records are retained in the audit with
status='quarantined' and are unavailable to numeric comparison.
"""
from __future__ import annotations
import math
import re
import unicodedata


def canonical_name(value: str) -> str:
    value = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', str(value))
    value = unicodedata.normalize('NFKC', value).replace('μ','u').replace('µ','u').casefold()
    return re.sub(r'[^a-z0-9]+',' ',value).strip()


# Every supported numeric attribute has a quantity type independent of its unit.
QUANTITIES = {
    'temperature': 'temperature', 'extrusion temperature':'temperature',
    'melt temperature':'temperature', 'barrel temperature':'temperature',
    'time':'duration', 'residence time':'duration',
    'pressure':'pressure', 'pressure at die':'pressure',
    'rotation speed':'rotation_rate', 'rotational frequency':'rotation_rate',
    'screw diameter length ratio':'screw_diameter_length_ratio',
    'aspect ratio':'aspect_ratio',
    'width':'length', 'length':'length', 'diameter':'length',
    'inner barrel diameter':'length', 'screw channel width':'length',
    'screw diameter':'length', 'screw length':'length', 'screw channel diameter':'length',
    'mass fraction':'mass_fraction', 'volume fraction':'volume_fraction',
    'density':'density', 'specific surface area':'specific_surface_area',
    'amperage':'electric_current', 'power':'power', 'voltage':'voltage',
    'torque':'torque', 'throughput':'mass_flow', 'amount':'amount', 'solvent amount':'amount',
}
CATEGORICAL = {'ambient condition','mixing method','equipment type','rotation mode',
               'extruder','additive','mixer','solvent name','chemical used','trade name','manufacturer or source name'}
NUMBER = re.compile(r'^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*(.*?)\s*$')
UNITS = {}
def register(quantity, canonical, aliases, factor=1., offset=0.):
    for alias in aliases:
        UNITS[(quantity, canonical_name(alias.replace('%',' percent ')))]=(canonical,factor,offset)
register('temperature','degC',['C','Celsius','degree Celsius','degrees Celsius','deg C','°C'])
register('temperature','degC',['K','Kelvin'] ,1.,-273.15)
register('temperature','degC',['F','Fahrenheit','°F'],5/9,-32*5/9)
register('duration','min',['min','mins','minute','minutes'])
register('duration','min',['s','sec','secs','second','seconds'],1/60)
register('duration','min',['h','hr','hrs','hour','hours'],60)
register('duration','min',['d','day','days'],1440)
register('duration','min',['week','weeks'],10080)
register('pressure','MPa',['MPa','megapascal','megapascals'])
register('pressure','MPa',['GPa','gigapascal'],1000)
register('pressure','MPa',['kPa','kilopascal','kilopascals'],.001)
register('pressure','MPa',['Pa','pascal','pascals'],.000001)
register('pressure','MPa',['bar','bars'],.1)
register('pressure','MPa',['psi'],.006894757293168)
register('rotation_rate','rpm',['rpm','revolutions per minute','rotation per minute'])
register('rotation_rate','rpm',['Hz','hertz'],60)
register('rotation_rate','rpm',['Radian per Minute','radians per minute','rad/min'],1/(2*math.pi))
register('length','nm',['nm','nanometer','nanometers','nanometre'])
register('length','nm',['um','micron','micrometer','micrometers','micrometre'],1000)
register('length','nm',['mm','millimeter','millimeters','millimetre'],1e6)
register('length','nm',['cm','centimeter'],1e7)
register('length','nm',['m','meter','metre'],1e9)
for q in ['mass_fraction','volume_fraction','aspect_ratio','screw_diameter_length_ratio']:
    register(q,'1',['','1','unitless','dimensionless'])
for q in ['mass_fraction','volume_fraction']:
    register(q,'1',['percent','%','Percent; %','wt %','vol %','weight percent','volume percent'],.01)
register('density','g/cm3',['Gram per Cubic Centimeter','g/cm3','g cm-3','g/cm^3'])
register('density','g/cm3',['Kilogram per Cubic Meter','kg/m3','kg/m^3'],.001)
register('specific_surface_area','m2/g',['Square Meter per Gram','m2/g','m^2/g'])
register('electric_current','A',['A','ampere','amperes'])
register('power','W',['W','watt','watts'])
register('voltage','V',['V','volt','volts'])
register('torque','N m',['N m','newton meter'])
register('mass_flow','kg/h',['kg/h','kilogram per hour'])
# 'amount' has no known physical kind until an explicit mass or volume unit is given.
register('amount','g',['g','gram','grams'])
register('amount','g',['kg','kilogram'],1000)
register('amount','mL',['mL','milliliter','millilitre'])
register('amount','mL',['L','liter','litre'],1000)


def normalize_numeric(key: str, raw: str, explicit_unit: str='') -> dict | None:
    name=canonical_name(key)
    # Numeric prefixes in chemical names (e.g. 1-decanethiol) are never numbers.
    if name in CATEGORICAL:
        return None
    m=NUMBER.fullmatch(str(raw))
    if m is None:
        return None
    value=float(m.group(1));suffix=m.group(2).strip();q=QUANTITIES.get(name)
    base={'raw':str(raw),'raw_unit':str(explicit_unit),'quantity_kind':q or 'unclassified'}
    def quarantine(reason):
        return {**base,'kind':'quarantined','value':None,'unit_group':None,
                'normalization_status':'quarantined','reason':reason}
    if not math.isfinite(value):return quarantine('nonfinite_value')
    if q is None:return quarantine('unknown_quantity_kind')
    raw_units=[u for u in [explicit_unit.strip(),suffix] if u]
    converted=[]
    for u in raw_units or ['']:
        unit_entry=UNITS.get((q,canonical_name(u.replace('%',' percent '))))
        if unit_entry is None:return quarantine('unrecognized_or_incompatible_unit' if u else 'unit_not_reported')
        canonical,factor,offset=unit_entry
        converted.append((canonical,value*factor+offset))
    if any(a[0]!=converted[0][0] or not math.isclose(a[1],converted[0][1],rel_tol=1e-12,abs_tol=1e-12) for a in converted):
        return quarantine('conflicting_unit_declarations')
    unit,val=converted[0]
    if q in {'mass_fraction','volume_fraction'} and not 0<=val<=1:return quarantine('fraction_outside_0_1')
    if q=='temperature' and val < -273.15:return quarantine('below_absolute_zero')
    if q in {'duration','pressure','rotation_rate','length','density','specific_surface_area','aspect_ratio','screw_diameter_length_ratio','amount'} and val<0:return quarantine('negative_quantity')
    if q in {'density','aspect_ratio','screw_diameter_length_ratio'} and val<=0:return quarantine('nonpositive_quantity')
    # Mass and volume amounts retain separate quantity types.
    typed_q=({'g':'mass_amount','mL':'volume_amount'}.get(unit,q) if q=='amount' else q)
    return {**base,'quantity_kind':typed_q,'kind':'numeric','value':val,
            'unit_group':f'{typed_q}|{unit}','canonical_unit':unit,
            'normalization_status':'accepted'}


def normalize_setting(key: str, raw: str, explicit_unit: str='') -> dict:
    name=canonical_name(key)
    if name in CATEGORICAL:
        return {'kind':'categorical','value':str(raw).strip(),'raw':str(raw),
                'raw_unit':explicit_unit,'normalization_status':'categorical',
                'quantity_kind':'categorical'}
    result=normalize_numeric(name,raw,explicit_unit)
    if result is not None:return result
    if name in QUANTITIES:
        return {'kind':'quarantined','value':None,'raw':str(raw),'raw_unit':explicit_unit,
                'unit_group':None,'quantity_kind':QUANTITIES[name],
                'normalization_status':'quarantined','reason':'unparsed_numeric_value'}
    return {'kind':'categorical','value':str(raw).strip(),'raw':str(raw),
            'raw_unit':explicit_unit,'normalization_status':'categorical','quantity_kind':'categorical'}
