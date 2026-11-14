"""One validated public-CLI pass per source state for the KV engine upgrade."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tomllib

import compare_aligners as shared


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def registry(lock):
    result = {}
    for package in tomllib.loads(lock)["package"]:
        if "source" in package:
            key = package["name"] + "@" + package["source"]
            result.setdefault(key, []).append([package["version"], package.get("checksum")])
    return {key: sorted(values) for key, values in result.items()}


def source_manifest(root, revision=None):
    if revision:
        names = subprocess.check_output(
            ["git", "ls-tree", "-r", "--name-only", revision, "--", "src", "Cargo.toml", "build.rs", ".cargo"],
            cwd=root, text=True,
        ).splitlines()
        return {name: hashlib.sha256(subprocess.check_output(
            ["git", "show", revision + ":" + name], cwd=root,
        ).replace(b"\r\n", b"\n")).hexdigest() for name in names}
    paths = [root / "Cargo.toml", root / "build.rs", *sorted((root / "src").rglob("*.rs"))]
    if (root / ".cargo").exists():
        paths += sorted(path for path in (root / ".cargo").rglob("*") if path.is_file())
    return {path.relative_to(root).as_posix(): hashlib.sha256(
        path.read_bytes().replace(b"\r\n", b"\n")
    ).hexdigest() for path in paths}


def reference_from_file(path):
    reference = {}
    current = None
    for line in path.read_bytes().splitlines():
        if line.startswith(b">"):
            current = line[1:].decode()
            reference[current] = bytearray()
        else:
            if current is None:
                raise ValueError("reference sequence precedes its FASTA header")
            reference[current].extend(line)
    return {name: bytes(sequence) for name, sequence in reference.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("before", "after"), required=True)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--reads", type=int, default=120000)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--new-candidate", action="store_true")
    args = parser.parse_args()
    if args.reads < 10 or args.reads % 10 or args.threads < 1:
        parser.error("reads must be a positive multiple of ten, at least ten; threads must be positive")
    root = Path(__file__).resolve().parent.parent
    work = root / "target/kv-upgrade"
    work.mkdir(parents=True, exist_ok=True)
    suffix = ".exe" if os.name == "nt" else ""
    binary = (args.binary or (work / ("before" + suffix) if args.phase == "before"
                             else root / "target/release" / ("kira_ls_aligner" + suffix))).resolve()
    result_file = work / (args.phase + "-results.json")
    env = {key: value for key, value in os.environ.items() if not key.startswith("KIRA_")}
    env["KIRA_STATS"] = "1"
    env["KIRA_BUILD_THREADS"] = str(args.threads)
    env["RAYON_NUM_THREADS"] = str(args.threads)
    config = {"reads_per_dataset": args.reads, "threads": args.threads, "seed": 20261026,
              "preset": "short", "only": "short", "batch_bases": 2000000,
              "environment": {key: env[key] for key in ("KIRA_STATS", "KIRA_BUILD_THREADS", "RAYON_NUM_THREADS")},
              "rustflags": env.get("RUSTFLAGS"), "encoded_rustflags": env.get("CARGO_ENCODED_RUSTFLAGS"),
              "processor": env.get("PROCESSOR_IDENTIFIER", platform.processor()), "platform": sys.platform}
    compiler = subprocess.check_output(["rustc", "-vV"], cwd=root, text=True)
    if args.phase == "before":
        revision = (work / "before-revision.txt").read_text().strip()
        lock = (work / "before.lock").read_text()
        saved_compiler = (work / "compiler.txt").read_text()
        if compiler.replace("\r\n", "\n").strip() != saved_compiler.replace("\r\n", "\n").strip():
            raise RuntimeError("baseline compiler receipt differs from current compiler")
    else:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip() + "+working-tree"
        lock = (root / "Cargo.lock").read_text()
    metadata = {"phase": args.phase, "revision": revision, "config": config,
                "binary_sha256": digest(binary), "compiler": compiler,
                "registry": registry(lock), "lock_sha256": hashlib.sha256(lock.encode()).hexdigest(),
                "source_manifest": source_manifest(root, revision if args.phase == "before" else None),
                "driver_sha256": digest(Path(__file__)), "shared_driver_sha256": digest(root / "tools/compare_aligners.py")}
    if result_file.exists():
        previous = json.loads(result_file.read_text())
        if args.phase == "after" and args.new_candidate:
            number = 1
            while (work / f"after-results.previous-{number}.json").exists():
                number += 1
            result_file.rename(work / f"after-results.previous-{number}.json")
        else:
            for field in ("config", "binary_sha256", "compiler", "source_manifest", "driver_sha256", "shared_driver_sha256"):
                if previous["metadata"][field] != metadata[field]:
                    raise RuntimeError("existing run belongs to different source/settings: " + field)
            print("Preserved existing results; no workload repeated:", result_file)
            if not previous.get("complete"):
                raise RuntimeError("previous state is incomplete; inspect saved rows before considering another run")
            return
    before = None
    if args.phase == "after":
        before = json.loads((work / "before-results.json").read_text())
        if not before.get("complete"):
            raise RuntimeError("baseline did not finish successfully")
        for field in ("config", "compiler", "driver_sha256", "shared_driver_sha256"):
            if before["metadata"][field] != metadata[field]:
                raise RuntimeError("paired comparison setting differs: " + field)
        old_registry = before["metadata"]["registry"]
        for name in old_registry.keys() & metadata["registry"].keys():
            if name.startswith("kira_kv_engine@"):
                continue
            if old_registry[name] != metadata["registry"][name]:
                raise RuntimeError("common dependency version/checksum differs: " + name)
        truth = json.loads((work / "truth.json").read_text())
        reference = reference_from_file(work / "reference.fa")
    else:
        reference, truth = shared.generate(work, args.reads)
    inputs = ["reference.fa", "truth.json", "single.r1.fq", "paired.r1.fq", "paired.r2.fq"]
    metadata["input_sha256"] = {name: digest(work / name) for name in inputs}
    if before and metadata["input_sha256"] != before["metadata"]["input_sha256"]:
        raise RuntimeError("paired input files differ")
    results = {"metadata": metadata, "rows": {}, "complete": False,
               "limitations": "One process per case/state, no warm-up/repeats; process peak includes runtime and I/O. Ambiguous repeat truth is reported without claiming perfect placement."}
    save(result_file, results)
    index = work / (args.phase + ".kiraidx")
    try:
        command = [binary, "index", work / "reference.fa", "-x", "short", "--only", "short", "-o", index]
        measurement = shared.run(command, work / ("index." + args.phase + ".log"), env)
        measurement["index_bytes"] = index.stat().st_size
        measurement["index_sha256"] = digest(index)
        results["rows"]["build"] = measurement
        save(result_file, results)
        print(args.phase, "build", measurement, flush=True)
        for dataset in ("single", "paired"):
            sam = work / (dataset + "." + args.phase + ".sam")
            command = [binary, "mem", "--index", index, work / "reference.fa", work / (dataset + ".r1.fq")]
            if dataset == "paired":
                command.append(work / "paired.r2.fq")
            command += ["-x", "short", "-t", str(args.threads), "-K", "2000000", "--no-PG", "-o", sam]
            measurement = shared.run(command, work / (dataset + "." + args.phase + ".log"), env)
            results["rows"][dataset] = measurement
            save(result_file, results)
            accuracy = shared.evaluate(sam, truth[dataset], reference)
            measurement["accuracy"] = accuracy
            save(result_file, results)
            if accuracy["invalid_cigar_nm"] != 0:
                raise RuntimeError(dataset + " has invalid CIGAR/NM records")
            if before and accuracy != before["rows"][dataset]["accuracy"]:
                raise RuntimeError(dataset + " truth metrics or full SAM-body digest changed")
            print(args.phase, dataset, measurement, flush=True)
        results["complete"] = True
        save(result_file, results)
    except Exception as error:
        results["error"] = {"type": type(error).__name__, "message": str(error)}
        save(result_file, results)
        raise
    if before:
        comparison = {}
        for name, current in results["rows"].items():
            old = before["rows"][name]
            comparison[name] = {metric: {"before": old[metric], "after": current[metric],
                                        "change_percent": (current[metric] / old[metric] - 1) * 100 if old[metric] else None}
                                for metric in ("wall_s", "cpu_s", "peak_mib") if old[metric] is not None and current[metric] is not None}
        save(work / "comparison.json", comparison)
        print(json.dumps(comparison, indent=2))
    print("All three public CLI cases passed:", result_file, flush=True)


if __name__ == "__main__":
    main()
