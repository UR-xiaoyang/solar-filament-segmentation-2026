"""A5-4 WP0: Freeze PQ contract and compute contract hash."""

import argparse
import hashlib
import json
from pathlib import Path

from .a5_pq_contract import (
    PQ_CONTRACT_VERSION,
    PQ_THRESHOLD,
    PQ_MIN_AREA,
    PQ_CONNECTIVITY,
    PQ_MAX_INSTANCES,
    PQ_MATCH_IOU,
    PQ_RELATION_IOU,
    CONTRACT_SCOPE,
)


def compute_contract_hash(contract_path: Path) -> str:
    """Compute SHA256 of contract document (excluding hash section)."""
    with open(contract_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Remove hash section if present
    lines = content.split("\n")
    filtered_lines = []
    skip_section = False

    for line in lines:
        if line.strip().startswith("## 9. Contract Hash"):
            skip_section = True
        elif line.strip().startswith("## 10.") or line.strip().startswith("## 11."):
            skip_section = False

        if not skip_section:
            filtered_lines.append(line)

    contract_content = "\n".join(filtered_lines).encode("utf-8")
    return hashlib.sha256(contract_content).hexdigest()


def freeze_contract(output_dir: Path) -> dict:
    """Generate frozen contract parameters and hash."""
    output_dir.mkdir(parents=True, exist_ok=True)

    contract_params = {
        "contract_version": PQ_CONTRACT_VERSION,
        "frozen_date": "2026-09-21",
        "component_generation": {
            "threshold": PQ_THRESHOLD,
            "min_area": PQ_MIN_AREA,
            "connectivity": PQ_CONNECTIVITY,
            "max_instances": PQ_MAX_INSTANCES,
        },
        "instance_matching": {
            "match_iou_threshold": PQ_MATCH_IOU,
            "relation_iou_threshold": PQ_RELATION_IOU,
            "algorithm": "greedy_descending_iou",
        },
        "primary_metric": "structural_error = FP + FN + 0.5*split + 0.5*merge",
        "secondary_metric": "matched_dice_mean",
        "scope": CONTRACT_SCOPE,
        "restrictions": [
            "development_oof_only",
            "no_test_labels",
            "no_temporal_holdout_selection",
            "immutable_artifacts",
        ],
    }

    # Write contract parameters
    contract_json = output_dir / "pq_contract.json"
    with open(contract_json, "w", encoding="utf-8") as f:
        json.dump(contract_params, f, indent=2)

    print(f"Wrote contract parameters: {contract_json}")

    # Compute contract document hash
    contract_doc = Path("artifacts/a5/a-5-4-wp0-pq-contract/PQ_CONTRACT.md")
    if contract_doc.exists():
        contract_hash = compute_contract_hash(contract_doc)
        contract_params["contract_document_sha256"] = contract_hash

        # Update JSON with hash
        with open(contract_json, "w", encoding="utf-8") as f:
            json.dump(contract_params, f, indent=2)

        print(f"Contract document hash: {contract_hash[:16]}...")
    else:
        print(f"WARNING: Contract document not found at {contract_doc}")

    # Write README
    readme = output_dir / "README.md"
    with open(readme, "w", encoding="utf-8") as f:
        f.write("# A5-4 WP0 PQ Contract (FROZEN)\n\n")
        f.write(f"**Version**: {PQ_CONTRACT_VERSION}\n")
        f.write(f"**Scope**: {CONTRACT_SCOPE}\n\n")
        f.write("## Frozen Parameters\n\n")
        f.write(f"- Threshold: {PQ_THRESHOLD}\n")
        f.write(f"- Min area: {PQ_MIN_AREA} pixels\n")
        f.write(f"- Match IoU: {PQ_MATCH_IOU}\n")
        f.write(f"- Primary metric: structural_error = FP + FN + 0.5×split + 0.5×merge\n\n")
        f.write("## Usage\n\n")
        f.write("```python\n")
        f.write("from filament_contract.a5_pq_contract import compute_pq_metrics\n\n")
        f.write("metrics = compute_pq_metrics(probability_map, ground_truth_masks, image_id)\n")
        f.write("print(f'Structural error: {metrics.structural_error:.4f}')\n")
        f.write("```\n\n")
        f.write("## Verification\n\n")
        f.write("All A5-4 artifacts must cite this contract in their `protocol.json`.\n")

    print(f"Wrote README: {readme}")

    print("\n" + "="*60)
    print("PQ CONTRACT FROZEN")
    print("="*60)
    print(f"Version:    {PQ_CONTRACT_VERSION}")
    print(f"Threshold:  {PQ_THRESHOLD}")
    print(f"Min area:   {PQ_MIN_AREA}")
    print(f"Match IoU:  {PQ_MATCH_IOU}")
    print(f"Scope:      {CONTRACT_SCOPE}")
    print("="*60)

    return contract_params


def main():
    parser = argparse.ArgumentParser(
        description="A5-4 WP0: Freeze PQ contract parameters and compute hash"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/a5/a-5-4-wp0-pq-contract"),
        help="Output directory for frozen contract",
    )

    args = parser.parse_args()
    freeze_contract(args.output_dir)


if __name__ == "__main__":
    main()
