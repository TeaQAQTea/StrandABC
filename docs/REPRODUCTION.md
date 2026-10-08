# StrandABC reproduction and compatibility

## Released checkpoints

The original release contains the following files under `strandabc/checkpoints/`:

| File | Variant | SHA-256 |
| --- | --- | --- |
| `full_epoch9.pt` | Full model | `241174896a7cb2dfdce4aa0313b9ccb5de6793dca0b0aeb205fb9d3dc09d59d9` |
| `ro_only.pt` | RO-seq only | `efa9a6af31f37a6c5dbb9130670097e4e98cea89d33dfff79fddcf1c5a8d1503` |
| `no_fusion.pt` | No strand fusion (`no_strand_fusion`), not no modality fusion | `d4db7321f8d09924472e4a77ef1ad41f7b09d433573c7895356d60ad2c07895f` |

These are the checkpoints used in the current final benchmark and ablation table.
This reorganization does not retrain models, change tensor names, or modify weights.
Checkpoint `args` retain historical paths as provenance, not portable input paths.
Supply the actual local FASTA and bigWig paths when running inference.

## Final evaluation protocol

- Train cells: K562, GM12878, HCT116 only.
- Train chromosomes: human autosomes excluding chr6 and chr8; validate on chr6.
- Seen-cell evaluation: chr8. Unseen-cell evaluation: chr8 and autosomes
  (the autosome set includes chr8).
- Input range: 1,650,000 bp; prediction grid: 2,200 bp; stride: 550,000 bp.
- Keep center 1,100,000 bp for evaluation; average overlapping predictions by
  total sum/count, independent of TSV chunk size.
- The comparison track/bar plots use matched 100 kb bins and exact bigWig
  target means. The ablation table uses the 2,200 bp prediction grid.
- Positive target is A (`score > 0`); zero and negative targets are B.
- ATAC-seq is for label selection/sign orientation only, not a model input.
- HepG2 uses GSM3393683 (GSE120105); hESC/H1 uses GSM1006729
  (GSE41009, 0 h); mESC uses GSE43390. A cell uses the same plus/minus
  RNA input pair across all variants and both evaluation scopes.
- Human assembly: hg38; mouse: mm10. H1/HepG2 targets are originally
  11 kb tracks sampled on the prediction grid, not native 2.2 kb targets.
- RNA source screening was exploratory, not an independent held-out selection.
  Point estimates do not establish uncertainty or across-seed stability.
- The full model is not best on every hESC ablation metric; preserve that result.

## Portable training configuration

See `strandabc/configs/train_three_cells.example.tsv` and
`strandabc/scripts/train_three_cells.sh`. Replace all placeholder paths before use.
Use the checkpoint manifest under `validation/checkpoints.json` to inspect
the actual saved training arguments. The example launcher reproduces the model
configuration, not a guarantee of bitwise-identical training across machines.

## Naming and compatibility

Use `from src.ab_compartment.strandabc import StrandABC` from the `strandabc/`
working directory. The old `CrossCompartment*` public class names and backbone
module import remain compatibility aliases. `BPLevelFusionRangePredictor` is
retained as an implementation name; it and `StrandABC` refer to the same class.

The reorganized repository is published as `TeaQAQTea/StrandABC`; the original
`rrrrraradis/StrandComp` repository is unchanged. The upstream Git history is
retained for auditing.

## Changes beyond naming

The 100 kb aggregation utility now uses sum/count across chunks instead of
mean-of-means and reads exact bigWig means. Historical metrics can change because
of these correctness fixes. The prediction wrapper refuses to overwrite an
existing prediction file. Model forward computations are unchanged.

## Validation

From the repository root:

```bash
/home/zwang/.conda/envs/torch25/bin/python scripts/validate_release.py
```

This checks all three checkpoint hashes and strict state-dict loading, compares
small synthetic CPU forward outputs with the original released implementation,
checks chunk-size-invariant aggregation and exact label means, and checks CLI
entrypoints. It is not a fresh genome-wide inference or a retraining run.
