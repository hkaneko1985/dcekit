"""Historical adapter, intentionally retains v0.1.0 defects for audit only."""
from collections import defaultdict
from material_similarity import DocumentedValue as V, MaterialInstance, ProcessStep, ReportingStatus

def role_value(row: dict, role: str, field: str) -> V:
    reported = role in row.get("reported_roles", [])
    names = row.get(field, [])
    if reported:
        return V.reported(names, kind="set")
    return V.unknown(kind="set")

def adapt_nanomine(row: dict) -> MaterialInstance:
    identity = {
        "matrix": role_value(row, "Matrix", "matrix_names"),
        "filler": role_value(row, "Filler", "filler_names"),
        "surface_treatment": role_value(row, "Surface Treatment", "surface_names"),
    }
    role_counts = defaultdict(int)
    for component in sorted(
        row.get("components", []), key=lambda item: (item.get("role", ""), item.get("name", ""))
    ):
        role = str(component.get("role", "unknown")).casefold().replace(" ", "_")
        index = role_counts[role]
        role_counts[role] += 1
        for attribute in component.get("attributes", []):
            if attribute.get("kind") != "numeric":
                continue
            attr_type = str(attribute.get("type", "value")).casefold()
            unit = attribute.get("unit_group") or attr_type
            if attr_type in {"massfraction", "volumefraction"}:
                unit = "fraction"
            identity[f"{role}#{index}:{attr_type}"] = V.reported(
                float(attribute["value"]), kind="numeric", unit_key=str(unit)
            )

    steps = []
    for raw_step in sorted(row.get("steps", []), key=lambda item: item.get("index", 0)):
        settings = {}
        for raw in raw_step.get("settings", []):
            if raw.get("kind") == "numeric":
                settings[raw["key"]] = V.reported(
                    float(raw["value"]),
                    kind="numeric",
                    unit_key=str(raw.get("unit_group") or raw["key"]),
                )
            else:
                settings[raw["key"]] = V.reported(str(raw["value"]), kind="categorical")
        steps.append(ProcessStep(str(raw_step["token"]), settings))

    return MaterialInstance(
        record_id=row["sample_id"],
        composition=V.unknown(kind="composition"),
        material_identity=identity,
        process_method=V.reported(row.get("process_families", []), kind="set"),
        process_steps=tuple(steps),
        process_sequence_status=ReportingStatus.REPORTED,
        context={"paper_group": row["paper_group"], "article_id": row["article_id"]},
    )
