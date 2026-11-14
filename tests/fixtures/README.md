# Legacy KV engine fixture

`kv_engine_v081.kiraidx` was generated from aligner revision `9115eec` using the
published `kira_kv_engine` 0.8.1 dependency, before the upgrade. It has a KIRAIDX4
outer container and the old engine hash identifier. SHA-256:
`8d015c0347d4d6e5bcb73b02b3a9cb70494bd6c7d297a8fa67f4dd0740724187`.

`kv_engine_v081.fa` contains only synthetic DNA: two 4,096-base LCG sequences
(seed `0x9e3779b97f4a7c15`, multiplier `6364136223846793005`, increment 1,
bases selected by bits 32..33), each followed by an ambiguous separator and a
repeated ACGTA suffix. No biological sample or personal data is included.

Recipe, with the old executable:

```text
kira_ls_aligner index tests/fixtures/kv_engine_v081.fa -o tests/fixtures/kv_engine_v081.kiraidx -x short --only short -k 7 -w 3 --max-occ 3
```

The public regression tests derive known buckets from the synthetic reference,
check occurrence order, true counts and truncation, then require byte-identical
resaving and working cloned mmap readers. The old executable is not required
at test time.
