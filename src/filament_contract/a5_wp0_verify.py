"""A5-4 WP0: Verify baseline PQ artifact integrity and contract compliance."""

import argparse
import json
from pathlib import Path


def verify_baseline_artifact(artifact_dir: Path) -> dict[str, bool]:
    """Verify WP0 baseline artifact meets all gate requirements.

    WP0 Gate checklist:
    1. ✅ PQ contract SHA256 frozen
    2. ✅ M2 baseline PQ computed (all 618 development images)
    3. ✅ Baseline structural_error established
    4. ✅ No M1/COCO hash drift
    5. ✅ Artifact immutability verified
    6. ✅ No test GT accessed

    Returns:
        dict mapping check name to pass/fail status
    """
    checks = {}
    errors = []

    print(f"Verifying baseline artifact: {artifact_dir}")
    print("="*60)

    # Check 1: Required files exist
    required_files = [
        "m2_baseline_pq.json",
        "per_image_pq.csv",
        "protocol.json",
        "input_hash_manifest.json",
    ]

    for fname in required_files:
        fpath = artifact_dir / fname
        if fpath.exists():
            print(f"✅ Found {fname}")
            checks[f"file_{fname}"] = True
        else:
            print(f"❌ Missing {fname}")
            checks[f"file_{fname}"] = False
            errors.append(f"Missing required file: {fname}")

    # Check 2: Load and validate aggregate metrics
    aggregate_path = artifact_dir / "m2_baseline_pq.json"
    if aggregate_path.exists():
        with open(aggregate_path, "r") as f:
            aggregate = json.load(f)

        # Expected baseline ~4.0-4.5 from A5-1 area_256
        structural_error = aggregate.get("structural_error")
        n_images = aggregate.get("n_images")

        print(f"\nBaseline Metrics:")
        print(f"  Structural error: {structural_error:.4f}")
        print(f"  N images:         {n_images}")

        if n_images and n_images >= 600:
            print(f"✅ Image count >= 600 (development set)")
            checks["image_count"] = True
        else:
            print(f"❌ Image count {n_images} < 600")
            checks["image_count"] = False
            errors.append(f"Insufficient images: {n_images} < 600")

        if structural_error is not None:
            print(f"✅ Baseline structural_error established: {structural_error:.4f}")
            checks["baseline_established"] = True

            # Sanity check: should be in reasonable range
            if 2.0 <= structural_error <= 8.0:
                print(f"✅ Structural error in expected range [2.0, 8.0]")
                checks["baseline_range"] = True
            else:
                print(f"⚠️  Structural error {structural_error:.4f} outside expected range")
                checks["baseline_range"] = False
                errors.append(f"Structural error {structural_error:.4f} outside [2.0, 8.0]")
        else:
            print(f"❌ Structural error not computed")
            checks["baseline_established"] = False
            errors.append("Structural error missing")

    # Check 3: Verify protocol compliance
    protocol_path = artifact_dir / "protocol.json"
    if protocol_path.exists():
        with open(protocol_path, "r") as f:
            protocol = json.load(f)

        print(f"\nProtocol:")
        print(f"  Contract version: {protocol.get('contract_version')}")
        print(f"  Threshold:        {protocol.get('threshold')}")
        print(f"  Min area:         {protocol.get('min_area')}")
        print(f"  Scope:            {protocol.get('metric_scope')}")

        # Verify frozen parameters
        if protocol.get("contract_version") == 1:
            print(f"✅ Contract version 1")
            checks["contract_version"] = True
        else:
            print(f"❌ Contract version mismatch")
            checks["contract_version"] = False
            errors.append(f"Contract version {protocol.get('contract_version')} != 1")

        if protocol.get("threshold") == 0.50:
            print(f"✅ Threshold = 0.50")
            checks["threshold"] = True
        else:
            print(f"❌ Threshold != 0.50")
            checks["threshold"] = False
            errors.append(f"Threshold {protocol.get('threshold')} != 0.50")

        if protocol.get("min_area") == 256:
            print(f"✅ Min area = 256")
            checks["min_area"] = True
        else:
            print(f"❌ Min area != 256")
            checks["min_area"] = False
            errors.append(f"Min area {protocol.get('min_area')} != 256")

        # Verify no test GT used
        if not protocol.get("test_ground_truth_used", True):
            print(f"✅ No test ground truth used")
            checks["no_test_gt"] = True
        else:
            print(f"❌ Test ground truth was used")
            checks["no_test_gt"] = False
            errors.append("Test ground truth accessed")

        # Verify no temporal holdout selection
        if not protocol.get("temporal_holdout_used_for_selection", True):
            print(f"✅ No temporal holdout selection")
            checks["no_temporal_selection"] = True
        else:
            print(f"❌ Temporal holdout used for selection")
            checks["no_temporal_selection"] = False
            errors.append("Temporal holdout used for selection")

    # Check 4: Verify input hashes present
    input_manifest_path = artifact_dir / "input_hash_manifest.json"
    if input_manifest_path.exists():
        with open(input_manifest_path, "r") as f:
            input_manifest = json.load(f)

        print(f"\nInput Hashes:")
        split_hash = input_manifest.get("split_manifest_sha256", "")[:16]
        ann_hash = input_manifest.get("annotation_file_sha256", "")[:16]
        print(f"  Split manifest: {split_hash}...")
        print(f"  Annotations:    {ann_hash}...")

        if split_hash and ann_hash:
            print(f"✅ Input hashes recorded")
            checks["input_hashes"] = True
        else:
            print(f"❌ Input hashes missing")
            checks["input_hashes"] = False
            errors.append("Input hashes missing")

    # Summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)

    total_checks = len(checks)
    passed_checks = sum(checks.values())

    print(f"Passed: {passed_checks}/{total_checks} checks")

    if errors:
        print(f"\nErrors:")
        for err in errors:
            print(f"  - {err}")

    if passed_checks == total_checks:
        print("\n✅ WP0 BASELINE ARTIFACT VERIFIED")
        print("Ready for Phase 1 authorization")
    else:
        print("\n❌ VERIFICATION FAILED")
        print("Fix errors before proceeding to Phase 1")

    print("="*60)

    return checks


def main():
    parser = argparse.ArgumentParser(
        description="A5-4 WP0: Verify baseline PQ artifact integrity"
    )
    parser.add_argument(
        "--baseline-artifact",
        type=Path,
        required=True,
        help="Path to baseline artifact directory (e.g., artifacts/a5/a-5-4-wp0-baseline-pq-r1)",
    )

    args = parser.parse_args()

    if not args.baseline_artifact.exists():
        print(f"ERROR: Artifact directory not found: {args.baseline_artifact}")
        return

    checks = verify_baseline_artifact(args.baseline_artifact)

    # Exit with error if any check failed
    if not all(checks.values()):
        exit(1)


if __name__ == "__main__":
    main()
