"""Compare trained checkpoints and measure CPU resolution trade-offs."""

import argparse
import hashlib
import json
import math
import platform
import statistics
import time
from pathlib import Path


def summarize_latencies(values):
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("Latencies must be finite positive numbers")
    ordered = sorted(values)
    return {
        "median_ms": statistics.median(ordered),
        "p95_ms": ordered[math.ceil(0.95 * len(ordered)) - 1],
    }


def choose_resolution(standard_ms, reduced_ms, standard_map, reduced_map, max_drop=0.02):
    values = (standard_ms, reduced_ms, standard_map, reduced_map, max_drop)
    if (
        not all(math.isfinite(value) for value in values)
        or min(standard_ms, reduced_ms) <= 0
        or not 0 <= standard_map <= 1
        or not 0 <= reduced_map <= 1
        or max_drop < 0
    ):
        raise ValueError("Invalid latency or mAP values")
    drop = standard_map - reduced_map
    faster = reduced_ms < standard_ms
    acceptable = drop <= max_drop + 1e-12
    return {
        "selected": "reduced" if faster and acceptable else "standard",
        "median_improved": faster,
        "map_drop": drop,
        "maximum_map_drop": max_drop,
        "quality_within_limit": acceptable,
    }


def make_selection(report, evidence_path):
    winner = report["model_selection"]["winner"]
    checkpoint = report["model_selection"]["comparison"][winner]
    resolution = report["results"][report["decision"]["selected"]]
    return {
        "experiment": winner,
        "checkpoint": checkpoint["checkpoint"],
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "selected_epoch": checkpoint["best_epoch"],
        "min_size": resolution["min_size"],
        "max_size": resolution["max_size"],
        "validation_map": resolution["metrics"]["map"],
        "evidence": str(evidence_path),
    }


def select_checkpoint(run_a: Path, run_b: Path):
    """Select by validation mAP after checking the controlled-run configuration."""
    histories = {}
    required = (
        "architecture",
        "initial_weights",
        "classes",
        "seed",
        "epochs",
        "batch_size",
        "learning_rate",
        "momentum",
        "weight_decay",
        "threads",
        "frozen",
        "train_image_ids",
        "manifest_sha256",
        "torch",
        "torchvision",
        "preprocessing",
        "validation_score_floor",
    )
    for name, root in (("A", run_a), ("B", run_b)):
        history = json.loads((root / "history.json").read_text(encoding="utf-8"))
        if history["config"]["experiment"] != name:
            raise ValueError(f"{root} does not contain experiment {name}")
        if [entry["epoch"] for entry in history["epochs"]] != list(
            range(1, history["config"]["epochs"] + 1)
        ):
            raise ValueError(f"Experiment {name} has not completed its full training budget")
        histories[name] = history
    a, b = histories["A"]["config"], histories["B"]["config"]
    mismatch = [key for key in required if key not in a or key not in b or a[key] != b[key]]
    if mismatch:
        raise ValueError(f"Experiments differ in controlled settings: {mismatch}")
    if a["augmentation"] != "none" or b["augmentation"] != "horizontal_flip(p=0.5)":
        raise ValueError("Expected no augmentation for A and horizontal flip for B")
    comparison = {}
    import torch

    for name, root in (("A", run_a), ("B", run_b)):
        history = histories[name]
        best = max(history["epochs"], key=lambda entry: entry["validation"]["metrics"]["map"])
        checkpoint = root / "best.pt"
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        metadata = payload["metadata"]
        if (
            metadata["epoch"] != best["epoch"]
            or metadata["config"] != history["config"]
            or abs(metadata["validation_metrics"]["map"] - best["validation"]["metrics"]["map"])
            > 1e-7
        ):
            raise ValueError(f"Checkpoint does not match experiment history: {checkpoint}")
        comparison[name] = {
            "best_epoch": best["epoch"],
            "validation_map": best["validation"]["metrics"]["map"],
            "validation_ap50": best["validation"]["metrics"]["map_50"],
            "checkpoint": checkpoint.as_posix(),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        }
    winner = "B" if comparison["B"]["validation_map"] > comparison["A"]["validation_map"] else "A"
    return winner, comparison, a["manifest_sha256"]


def main():
    import torch
    import torchvision

    from pedestrian.app_support import matches_manifest_hash
    from pedestrian.data import PennFudanDataset
    from pedestrian.evaluate import configure_evaluation, evaluate_model
    from pedestrian.train import load_checkpoint

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("manifests/splits.json"))
    parser.add_argument("--run-a", type=Path, required=True)
    parser.add_argument("--run-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection-output", type=Path, help="Optional new selection JSON path")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.output.exists():
        parser.error("Output already exists; choose a new output file")
    if args.selection_output and args.selection_output.exists():
        parser.error("Selection output already exists; choose a new output file")
    if args.selection_output and args.selection_output.resolve() == args.output.resolve():
        parser.error("Selection output and report output must be different files")
    torch.set_num_threads(args.threads)
    winner, comparison, manifest_hash = select_checkpoint(args.run_a, args.run_b)
    if not matches_manifest_hash(args.manifest, manifest_hash):
        raise ValueError("Manifest differs from the one used for training")
    model, metadata = load_checkpoint(comparison[winner]["checkpoint"])
    if metadata["epoch"] != comparison[winner]["best_epoch"]:
        raise ValueError("Checkpoint epoch does not match the selected history")
    model = configure_evaluation(model)
    dataset = PennFudanDataset(args.dataset_root, args.manifest, "validation")
    ids = [i for i in dataset.image_ids if i.startswith("FudanPed")][:5]
    ids += [i for i in dataset.image_ids if i.startswith("PennPed")][:5]
    if len(ids) != 10:
        raise ValueError("Need at least five validation images from each source")
    benchmark_data = PennFudanDataset(args.dataset_root, args.manifest, "validation", image_ids=ids)
    images = [benchmark_data[i][0] for i in range(len(benchmark_data))]
    results = {}
    for name, minimum, maximum in (("standard", 320, 640), ("reduced", 256, 512)):
        model.transform.min_size = (minimum,)
        model.transform.max_size = maximum
        print(f"Evaluating {name} resolution on validation...", flush=True)
        evaluation = evaluate_model(model, dataset, progress=False)
        if (
            name == "standard"
            and abs(evaluation["metrics"]["map"] - comparison[winner]["validation_map"]) > 1e-6
        ):
            raise RuntimeError("Checkpoint validation mAP does not match the selected training run")
        durations = []
        with torch.inference_mode():
            for i in range(5):
                model([images[i]])
            for i in range(30):
                started = time.perf_counter()
                model([images[i % len(images)]])
                durations.append((time.perf_counter() - started) * 1000)
        results[name] = {
            "min_size": minimum,
            "max_size": maximum,
            "metrics": evaluation["metrics"],
            "counts_at_confidence_0_5": evaluation["counts"],
            "latencies_ms": durations,
            **summarize_latencies(durations),
        }
        print(
            f"{name}: mAP {results[name]['metrics']['map']:.4f}, "
            f"median {results[name]['median_ms']:.1f} ms",
            flush=True,
        )
    standard, reduced = results["standard"], results["reduced"]
    decision = choose_resolution(
        standard["median_ms"],
        reduced["median_ms"],
        standard["metrics"]["map"],
        reduced["metrics"]["map"],
    )
    report = {
        "model_selection": {"winner": winner, "comparison": comparison},
        "split": "validation",
        "manifest_sha256": manifest_hash,
        "benchmark_image_ids": ids,
        "warmup_predictions": 5,
        "measured_predictions": 30,
        "timing_boundary": (
            "model call on preloaded CPU tensor, including internal resize, inference and "
            "postprocessing; excludes file decoding and drawing"
        ),
        "device": "cpu",
        "threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__),
        "results": results,
        "decision": decision,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.selection_output:
        args.selection_output.parent.mkdir(parents=True, exist_ok=True)
        args.selection_output.write_text(
            json.dumps(make_selection(report, args.output.as_posix()), indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps({"winner": winner, "decision": decision}, indent=2))


if __name__ == "__main__":
    main()
