"""Reproducible, truth-labelled before/after benchmark (Python stdlib only).

Build/copy the unmodified release binary before editing Rust sources. Example:
python tools/compare_aligners.py --before target/comparison/before.exe \
    --after target/release/kira_ls_aligner.exe --reads 60000 --repeats 5
Generated data, SAM, logs and raw measurements stay under target/comparison.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import random
import re
import statistics
import subprocess
import time


def dna(rng, n):
    return bytes(rng.choices(b"ACGT", k=n))


def revcomp(seq):
    return seq.translate(bytes.maketrans(b"ACGT", b"TGCA"))[::-1]


def generate(root, count):
    rng = random.Random(20261026)
    ref = {"unique": dna(rng, 500_000)}
    ancestor = dna(rng, 25_000)
    for copy in range(12):
        seq = bytearray(ancestor)
        for pos in rng.sample(range(len(seq)), 125 + copy * 10):
            seq[pos] = rng.choice([b for b in b"ACGT" if b != seq[pos]])
        ref[f"paralog{copy}"] = bytes(seq)
    # Complete moderately frequent buckets exercise adaptive occurrence sampling.
    repeat = dna(rng, 900)
    for copy in range(40):
        ref[f"repeat{copy}"] = dna(rng, 150) + repeat + dna(rng, 150)
    with (root / "reference.fa").open("wb") as out:
        for name, seq in ref.items():
            out.write(b">" + name.encode() + b"\n" + seq + b"\n")
    truth = {}
    for dataset, paired in [("single", False), ("paired", True)]:
        dataset_truth = {}
        r1 = (root / f"{dataset}.r1.fq").open("wb")
        r2 = (root / f"{dataset}.r2.fq").open("wb") if paired else None
        for i in range(count // (2 if paired else 1)):
            group = ["exact", "snp", "indel", "paralog", "repeat"][i % 5]
            if group == "paralog":
                name = f"paralog{rng.randrange(12)}"
            elif group == "repeat":
                name = f"repeat{rng.randrange(40)}"
            else:
                name = "unique"
            seq = ref[name]
            start = rng.randrange(20, len(seq) - 520)
            fragment = rng.randrange(320, 480)
            qname = f"{dataset}{i}:{name}:{start}-{start + fragment}"
            for role in ([1, 2] if paired else [0]):
                pos = start if role != 2 else start + fragment - 150
                bases = bytearray(seq[pos:pos + 150])
                event = None
                if group == "indel":
                    length = rng.choice([1, 2, 3, 5, 8, 12])
                    at = rng.choice([5, 10, 30, 65, 100, 130, 143])
                    if i % 2:
                        bases[at:at] = dna(rng, length)
                        event = ["I", pos + at, length]
                    else:
                        del bases[at:at + length]
                        event = ["D", pos + at, min(length, 150 - at)]
                if group in ("snp", "paralog") or (group == "indel" and i % 3 == 0):
                    for at in rng.sample(range(len(bases)), 1 + i % 3):
                        bases[at] = rng.choice([b for b in b"ACGT" if b != bases[at]])
                reverse = role == 2 or (not paired and i % 2 == 1)
                bases = revcomp(bytes(bases)) if reverse else bytes(bases)
                out = r2 if role == 2 else r1
                out.write(b"@" + qname.encode() + b"\n" + bases + b"\n+\n" + b"I" * len(bases) + b"\n")
                dataset_truth[f"{qname}/{role}"] = [name, pos, reverse, group, event]
        r1.close()
        if r2:
            r2.close()
        truth[dataset] = dataset_truth
    (root / "truth.json").write_text(json.dumps(truth), encoding="utf-8")
    return ref, truth


class MemoryCounters(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong)] + [
        (name, ctypes.c_size_t) for name in (
            "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
            "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
            "PagefileUsage", "PeakPagefileUsage")]


def run(command, log, env):
    start = time.perf_counter()
    with log.open("wb") as out:
        process = subprocess.Popen([str(x) for x in command], stdout=out, stderr=out, env=env)
        peak = 0
        while process.poll() is None:
            if os.name == "nt":
                counters = MemoryCounters()
                counters.cb = ctypes.sizeof(counters)
                if ctypes.windll.psapi.GetProcessMemoryInfo(
                        ctypes.c_void_p(int(process._handle)), ctypes.byref(counters), counters.cb):
                    peak = max(peak, counters.PeakWorkingSetSize)
            time.sleep(0.005)
        wall = time.perf_counter() - start
        if process.returncode:
            raise RuntimeError(f"{command}: exit {process.returncode}\n{log.read_text(errors='replace')[-4000:]}")
    cpu = None
    if os.name == "nt":
        created, exited, kernel, user = (ctypes.c_ulonglong() for _ in range(4))
        if ctypes.windll.kernel32.GetProcessTimes(ctypes.c_void_p(int(process._handle)),
                ctypes.byref(created), ctypes.byref(exited), ctypes.byref(kernel), ctypes.byref(user)):
            cpu = (kernel.value + user.value) / 10_000_000
    return {"wall_s": wall, "cpu_s": cpu, "peak_mib": peak / (1024 ** 2)}


def evaluate(path, truth, reference):
    metrics = {"reads": len(truth), "mapped": 0, "correct": 0, "wrong": 0,
               "wrong_mapq30": 0, "indel_reads": 0, "indel_recovered": 0,
               "invalid_cigar_nm": 0}
    groups = {}
    seen = set()
    digest = hashlib.sha256()
    for line in path.open(encoding="utf-8"):
        if line.startswith("@"):
            continue
        digest.update(line.encode())
        f = line.rstrip().split("\t")
        flag = int(f[1])
        if flag & (0x100 | 0x800):
            continue
        role = 1 if flag & 0x40 else 2 if flag & 0x80 else 0
        key = f"{f[0]}/{role}"
        if key in seen:
            raise AssertionError(f"duplicate primary: {key}")
        seen.add(key)
        name, start, reverse, group, event = truth[key]
        g = groups.setdefault(group, {"reads": 0, "correct": 0, "wrong": 0})
        g["reads"] += 1
        if event:
            metrics["indel_reads"] += 1
        if flag & 4:
            continue
        metrics["mapped"] += 1
        pos = int(f[3]) - 1
        correct = f[2] == name and abs(pos - start) <= 5 and bool(flag & 16) == reverse
        metrics["correct" if correct else "wrong"] += 1
        g["correct" if correct else "wrong"] += 1
        if not correct and int(f[4]) >= 30:
            metrics["wrong_mapq30"] += 1
        qi, ri, nm, recovered = 0, pos, 0, False
        sequence = f[9].encode()
        ref = reference[f[2]]
        try:
            ops = re.findall(r"(\d+)([MIDNSH=X])", f[5])
            assert "".join(n + op for n, op in ops) == f[5]
            for n, op in ops:
                n = int(n)
                if event and correct and op == event[0] and n == event[2] and abs(ri - event[1]) <= 5:
                    recovered = True
                if op in "M=X":
                    assert ri + n <= len(ref) and qi + n <= len(sequence)
                    nm += sum(a != b for a, b in zip(sequence[qi:qi+n], ref[ri:ri+n]))
                    qi += n
                    ri += n
                elif op == "I":
                    qi += n
                    nm += n
                elif op == "D":
                    ri += n
                    nm += n
                elif op == "S":
                    qi += n
                elif op == "N":
                    ri += n
            assert qi == len(sequence)
            reported_nm = next(int(x[5:]) for x in f[11:] if x.startswith("NM:i:"))
            assert nm == reported_nm
        except (AssertionError, StopIteration, IndexError):
            metrics["invalid_cigar_nm"] += 1
        metrics["indel_recovered"] += int(recovered)
    assert len(seen) == len(truth), (len(seen), len(truth))
    metrics["groups"] = groups
    metrics["sam_body_sha256"] = digest.hexdigest()
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path)
    parser.add_argument("--reads", type=int, default=60000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("target/comparison"))
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    ref, truth = generate(root, args.reads)
    env = {k: v for k, v in os.environ.items() if not k.startswith("KIRA_")}
    env["KIRA_STATS"] = "1"
    # Identical index bytes for both arms; no index construction in timed runs.
    index = root / "reference.kiraidx"
    run([args.before.resolve(), "index", root / "reference.fa", "-o", index], root / "index.log", env)
    binaries = {"before": args.before.resolve()}
    if args.after:
        binaries["after"] = args.after.resolve()
    results = {"config": {"reads_per_dataset": args.reads, "repeats": args.repeats,
                          "threads": args.threads, "seed": 20261026}, "runs": {}}
    for dataset in truth:
        arms = {arm: [] for arm in binaries}
        for iteration in range(-1, args.repeats):  # untimed warm-up, then alternating AB/BA
            order = list(binaries) if iteration % 2 == 0 else list(reversed(binaries))
            for arm in order:
                sam = root / f"{dataset}.{arm}.sam"
                command = [binaries[arm], "mem", "--index", index, root / "reference.fa",
                           root / f"{dataset}.r1.fq"]
                if dataset == "paired":
                    command.append(root / f"{dataset}.r2.fq")
                command += ["-t", str(args.threads), "-K", "2000000", "-o", sam]
                measurement = run(command, root / f"{dataset}.{arm}.{iteration}.log", env)
                if iteration >= 0:
                    digest = hashlib.sha256()
                    with sam.open("rb") as stream:
                        for line in stream:
                            if not line.startswith(b"@"):
                                digest.update(line)
                    measurement["sam_body_sha256"] = digest.hexdigest()
                    arms[arm].append(measurement)
                print(dataset, iteration, arm, measurement, flush=True)
        results["runs"][dataset] = {}
        for arm, measurements in arms.items():
            results["runs"][dataset][arm] = {
                "measurements": measurements,
                "median_wall_s": statistics.median(x["wall_s"] for x in measurements),
                "median_cpu_s": statistics.median(x["cpu_s"] for x in measurements) if os.name == "nt" else None,
                "median_peak_mib": statistics.median(x["peak_mib"] for x in measurements),
                "accuracy": evaluate(root / f"{dataset}.{arm}.sam", truth[dataset], ref),
                "binary_sha256": hashlib.sha256(binaries[arm].read_bytes()).hexdigest(),
                "deterministic_across_runs": len({x["sam_body_sha256"] for x in measurements}) == 1,
            }
        (root / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
