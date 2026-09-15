"""Audited NanoMine adapter with source traceability and unordered multi-values."""
from __future__ import annotations
from collections import defaultdict
from .schema import DocumentedValue as V, MaterialInstance, ProcessStep, ReportingStatus
from .normalization import canonical_name, normalize_setting, normalize_numeric


def documented_group(items):
    """Keep duplicate names as a multiset; never invent physical zone indices."""
    accepted=[x for x in items if x.get('kind') in {'numeric','categorical'}]
    if not accepted:return V.unknown()
    # A partly quarantined collection is not silently treated as complete.
    if len(accepted)!=len(items):return V.unknown()
    kinds={x['kind'] for x in accepted};units={x.get('unit_group') for x in accepted}
    if len(kinds)!=1 or len(units)!=1:return V.unknown()
    kind=accepted[0]['kind'];unit=accepted[0].get('unit_group')
    values=[float(x['value']) if kind=='numeric' else str(x['value']) for x in accepted]
    if len(values)==1:return V.reported(values[0],kind=kind,unit_key=unit)
    return V.reported(tuple(sorted(values)),kind=kind+'_multiset',unit_key=unit)


def adapt_nanomine(row, *, include_component_descriptors=False):
    identity={}
    for role,field,name in [('Matrix','matrix_names','matrix'),('Filler','filler_names','filler'),('Surface Treatment','surface_names','surface_treatment')]:
        identity[name]=V.reported(row.get(field,[]),kind='set') if role in row.get('reported_roles',[]) else V.unknown(kind='set')
    counts=defaultdict(int)
    for component in sorted(row.get('components',[]),key=lambda x:(x.get('role',''),x.get('name',''))):
        role=canonical_name(component.get('role','unknown')).replace(' ','_');idx=counts[role];counts[role]+=1
        grouped=defaultdict(list)
        for attr in component.get('attributes',[]):
            attrname=canonical_name(attr.get('type',''))
            if not include_component_descriptors and attrname not in {'mass fraction','volume fraction'}:continue
            value=normalize_numeric(attrname,str(attr.get('raw_value',attr.get('raw',''))),str(attr.get('raw_unit','')))
            if value is not None:grouped[attrname].append(value)
        for key,values in grouped.items():identity[f'{role}#{idx}:{key}']=documented_group(values)
    steps=[]
    for source_step in sorted(row.get('steps',[]),key=lambda x:(x.get('index',0),x.get('uri',''))):
        groups=defaultdict(list);provenance=defaultdict(list)
        for position,item in enumerate(source_step.get('settings',[])):
            key=canonical_name(item['key'])
            parsed=normalize_setting(key,str(item.get('raw',item.get('value',''))),str(item.get('raw_unit','')))
            groups[key].append(parsed)
            provenance[key].append({'source_position':position,'raw':str(item.get('raw',item.get('value',''))),
                                    'raw_unit':str(item.get('raw_unit','')),'normalization':parsed,
                                    'correspondence':'unordered values; physical zone identity not supplied'})
        values={key:documented_group(items) for key,items in groups.items()}
        steps.append(ProcessStep(str(source_step['token']),values,source_step.get('uri'),dict(provenance)))
    return MaterialInstance(record_id=row['sample_id'],composition=V.unknown(kind='composition'),
        material_identity=identity,process_method=V.reported(row.get('process_families',[]),kind='set'),
        process_steps=tuple(steps),process_sequence_status=ReportingStatus.REPORTED,
        context={'paper_group':row['paper_group'],'article_id':row['article_id'],'source_doi':row.get('doi')})
