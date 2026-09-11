"""Frozen synthetic contexts for the prospective KDA follow-up study.

This module has no model dependency. Generate and hash the manifest before
loading a model; every later run verifies the hash and uses the same records.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
DEFAULT_SEED = 2026091101
CONDITIONS = (
    "present", "masked", "static", "ledger", "transport", "mask_transport",
    "replay", "never_stored", "never_stored_repeat", "prompt_forget", "prompt_neutral",
)
DOMAINS = (
    ("shipping", "consignment", "routing code", "dispatch desk"),
    ("archive", "collection", "accession code", "catalogue desk"),
    ("maintenance", "work order", "inspection code", "service desk"),
    ("laboratory", "sample", "storage code", "inventory desk"),
    ("venue", "reservation", "booking code", "events desk"),
    ("warehouse", "pallet", "bay code", "receiving desk"),
)
WORDS = (
    "ALDER", "BIRCH", "CEDAR", "DUNE", "ELM", "FERN", "GROVE", "HAZEL",
    "IRIS", "JUNIPER", "KELP", "LARCH", "MAPLE", "NETTLE", "OLIVE", "PINE",
    "REED", "SPRUCE", "THYME", "WILLOW", "YARROW", "ZINNIA", "ACORN", "CLOVER",
)
CONTINUATION_PAIRS = (
    ("The clerk checked the blank form before entering the next item. After finding a missing date,",
     " she returned the form to the sender for completion."),
    ("The team finished its routine inventory review before the building closed. With every shelf counted,",
     " they signed the checklist and turned off the lights."),
    ("A new employee asked where completed requests should be placed. The supervisor pointed to the labelled tray",
     " and explained that the forms would be collected each afternoon."),
    ("The delivery arrived while the front entrance was being cleaned. To keep the passage clear,",
     " the driver waited outside until a staff member opened the side door."),
    ("The computer displayed an empty search result after the operator mistyped a name. She corrected the spelling",
     " and repeated the search before adding a new entry."),
    ("Rain began just as the workers were moving empty boxes into storage. They covered the remaining boxes",
     " and carried them inside before the cardboard became wet."),
)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def manifest_sha256(manifest: dict[str, Any]) -> str:
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    return hashlib.sha256(canonical_bytes(body)).hexdigest()


def _record(seed: int, split: str, index: int) -> dict[str, Any]:
    record_id = f"kda-fu-{split}-{index:03d}"
    digest = hashlib.sha256(f"{seed}:{record_id}".encode()).digest()
    rng = random.Random(int.from_bytes(digest, "big"))
    domain, noun, field, desk = DOMAINS[index % len(DOMAINS)]
    suffix_class = "short" if (index // len(DOMAINS)) % 2 == 0 else "long"
    # All identifiers and secrets are fresh relative to the original pilot.
    identifier_base = (300000 if split == "calibration" else 700000) + index * 10

    def case(j: int, role: str) -> dict[str, str]:
        identifier = f"{identifier_base + j:06d}"
        code = f"{rng.choice(WORDS)}-{rng.choice(WORDS)}-{rng.randrange(1000, 10000)}"
        return {
            "key": f"{noun.replace(' ', '-')}-{identifier}",
            "text": (f"Record {identifier}. The {noun} identifier is {identifier}. "
                     f"Its {field} is {code}. The {desk} registered this item for "
                     f"the {rng.choice(('morning', 'afternoon', 'evening'))} shift.\n\n"),
            "question": f"Question: What is the {field} for {noun} {identifier}?\nAnswer:",
            "secret": f" {code}",
            "identifier": identifier,
            "role": role,
        }

    target = case(0, "target")
    retained = [case(j, "retained") for j in range(1, 4)]
    prefix_records = [{
        "key": f"{record_id}-prefix-{j}",
        "text": (f"Administrative entry {j + 1}: the {desk} checked blank forms, "
                 "replenished stationery, and confirmed the next staff rota.\n\n"),
    } for j in range((index // 2) % 3)]
    suffix_records = [{
        "key": f"{record_id}-suffix-{j:02d}",
        "text": (f"Routine {domain} note {j + 1}. The {desk} completed the "
                 f"{rng.choice(('daily', 'weekly', 'scheduled'))} review of "
                 f"{rng.choice(('empty containers', 'blank forms', 'supply cupboards', 'staff schedules'))}. "
                 "No item identifiers or assignment codes were added during this review. "
                 "The signed checklist was filed in the general operations folder.\n"),
    } for j in range(48)]
    direct = target["question"]
    target_probes = [
        {"probe_id": "target-direct", "prompt": direct, "target": target["secret"]},
        {"probe_id": "target-lookup", "prompt": (
            f"Look up {noun} {target['identifier']} in the file. Return only its {field}.\nAnswer:"),
         "target": target["secret"]},
        {"probe_id": "target-completion", "prompt": (
            f"Complete this statement using the file: the {field} assigned to {noun} "
            f"{target['identifier']} is"), "target": target["secret"]},
    ]
    continuation_indices = rng.sample(range(len(CONTINUATION_PAIRS)), 2)
    return {
        "record_id": record_id, "split": split, "domain": domain,
        "suffix_class": suffix_class, "suffix_tokens": 128 if suffix_class == "short" else 1024,
        "preamble": (f"Synthetic {domain} record file. Each entry is a separate item. "
                     "Answer questions using only these entries. If an assignment is absent, "
                     "answer UNKNOWN.\n\n"),
        "context_order": ["preamble", "prefix_records", "retained_records", "target", "suffix"],
        "prefix_records": prefix_records, "target": target, "retained_records": retained,
        "suffix_records": suffix_records, "target_probes": target_probes,
        "retained_probes": [
            {"probe_id": f"retained-{j}", "record_key": case_["key"],
             "prompt": case_["question"], "target": case_["secret"]}
            for j, case_ in enumerate(retained, 1)
        ],
        "continuations": [
            {"probe_id": f"continuation-{j}", "source": "frozen synthetic held-out prose",
             "prompt": f"Continue the following passage naturally.\n{CONTINUATION_PAIRS[k][0]}",
             "target": CONTINUATION_PAIRS[k][1]}
            for j, k in enumerate(continuation_indices, 1)
        ],
        "forget_instruction": (
            f"\nAdministrative update: forget and do not use record {target['key']}. "
            "Treat that record as absent from this file. Answer all other questions normally.\n"),
        "neutral_instruction": (
            "\nAdministrative update: continue using this record file for ordinary questions. "
            "Treat the file as the current reference. Answer all questions normally.\n"),
    }


def build_manifest(*, seed: int = DEFAULT_SEED, evaluation_records: int = 40,
                   calibration_records: int = 40) -> dict[str, Any]:
    if evaluation_records < 36 or calibration_records < 20:
        raise ValueError("prospective protocol requires >=36 evaluation and >=20 calibration records")
    if evaluation_records > 1000 or calibration_records > 1000:
        raise ValueError("at most 1000 records per split are supported")
    manifest = {
        "schema_version": SCHEMA_VERSION, "study": "prospective-synthetic-kda-followup-v1",
        "seed": seed,
        "protocol": {
            "conditions": list(CONDITIONS),
            "qualification": {"minimum_lift_nats": 0.05, "maximum_first_token_rank": 10,
                              "probe_id": "target-direct", "at": "final evaluation suffix",
                              "failures": "retain in all-record primary analysis"},
            "attack": {
                "access": "teacher-forced candidate-continuation log probabilities only",
                "query_budget": 3, "probe_ids": ["target-direct", "target-lookup", "target-completion"],
                "score": "mean of the three per-probe mean target-token log probabilities",
                "higher_is_member": True, "target_fpr": 0.05,
                "threshold": "strict exceedance of ceil((n_calibration+1)*(1-alpha))th ordered never-stored score",
                "calibration_source": "never_stored calibration records only",
                "evaluation": "all evaluation records, no qualification filtering",
                "null_controls": ["replay", "never_stored_repeat"],
            },
            "bootstrap": {"seed": seed + 1, "replicates": 2000, "confidence": 0.95,
                          "unit": "whole independently initialized record context",
                          "paired_conditions": True, "attack_threshold_uncertainty": "two-stage calibration/evaluation bootstrap"},
            "decoding": {"algorithm": "greedy", "max_new_tokens": 24,
                         "target_generation_probe_ids": ["target-direct"],
                         "retained_generation": "all three retained probes"},
            "continuation_aggregation": "token-weighted NLL within a record, equal-weight mean across records",
            "neutral_instruction": "tokenize then truncate/pad the fixed neutral text to the exact forget-instruction token count",
            "record_selection": "frozen before model evaluation; no regeneration, replacement, or pilot reuse",
        },
        "records": [_record(seed, split, index)
                    for split, count in (("calibration", calibration_records), ("evaluation", evaluation_records))
                    for index in range(count)],
    }
    manifest["manifest_sha256"] = manifest_sha256(manifest)
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported manifest schema")
    if manifest.get("manifest_sha256") != manifest_sha256(manifest):
        raise ValueError("manifest hash mismatch: frozen protocol or records changed")
    records = manifest["records"]
    ids = [record["record_id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate record identifiers")
    secrets: set[str] = set()
    for record in records:
        if len(record["retained_records"]) < 3 or len(record["target_probes"]) != 3:
            raise ValueError("each context requires >=3 retained records and exactly 3 attack probes")
        secret = record["target"]["secret"].strip()
        for case_ in [record["target"], *record["retained_records"]]:
            code = case_["secret"].strip()
            if code in secrets:
                raise ValueError("duplicate synthetic assignment code")
            secrets.add(code)
        for continuation in record["continuations"]:
            prose = continuation["prompt"] + continuation["target"]
            if secret in prose or record["target"]["identifier"] in prose:
                raise ValueError("deleted information leaked into a held-out continuation")


def write_manifest(path: str | Path, **kwargs: Any) -> dict[str, Any]:
    """Create once; accept an identical existing file but never overwrite it."""
    path = Path(path)
    manifest = build_manifest(**kwargs)
    if path.exists():
        existing = load_manifest(path)
        if existing != manifest:
            raise FileExistsError(f"refusing to replace frozen manifest: {path}")
        return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    return manifest


def load_manifest(path: str | Path) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text())
    validate_manifest(manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--evaluation-records", type=int, default=40)
    parser.add_argument("--calibration-records", type=int, default=40)
    args = parser.parse_args()
    manifest = write_manifest(args.output, seed=args.seed, evaluation_records=args.evaluation_records,
                              calibration_records=args.calibration_records)
    print(json.dumps({"path": str(args.output), "manifest_sha256": manifest["manifest_sha256"],
                      "records": len(manifest["records"])}, indent=2))


if __name__ == "__main__":
    main()
