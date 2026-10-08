#!/usr/bin/env python
"""Retained release audit: renamed model equivalence and aggregation regression."""
from pathlib import Path
from types import SimpleNamespace, ModuleType
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'strandabc'
sys.path.insert(0, str(PACKAGE))
sys.path.insert(0, str(PACKAGE / 'scripts'))
import numpy as np
import pandas as pd
import torch
import pyBigWig
import predict_ranges
import aggregate_predictions as aggregate
from src.ab_compartment.strandabc import StrandABC
from src.ab_compartment import models

EXPECTED = {
    'full_epoch9.pt': '241174896a7cb2dfdce4aa0313b9ccb5de6793dca0b0aeb205fb9d3dc09d59d9',
    'ro_only.pt': 'efa9a6af31f37a6c5dbb9130670097e4e98cea89d33dfff79fddcf1c5a8d1503',
    'no_fusion.pt': 'd4db7321f8d09924472e4a77ef1ad41f7b09d433573c7895356d60ad2c07895f',
}


def main():
    torch.set_num_threads(4)
    torch.manual_seed(43)
    destination = ROOT / 'validation'
    destination.mkdir(exist_ok=True)
    assert StrandABC is models.BPLevelFusionRangePredictor
    assert models.CrossCompartmentFusionRangePredictor is models.StrandABCFusionRangePredictor
    revision = '37764510682c8631add06ce5b16c69ae1c48757a'
    # Keep the original implementation in the bundle to make the test independent
    # of local Git metadata after packaging. First export is from the exact upstream.
    reference_path = destination / 'upstream_models_reference.py'
    if not reference_path.exists():
        original = subprocess.check_output(
            ['git', 'show', f'{revision}:CrossCompartment/src/ab_compartment/models.py'],
            cwd=ROOT, text=True)
        reference_path.write_text(original)
    original = reference_path.read_text()
    module = ModuleType('upstream_reference')
    exec(compile(original, str(reference_path), 'exec'), module.__dict__)
    checks = []
    dummy = SimpleNamespace(range_size=1650000, bin_size=2200, input_length=1024)
    for filename, expected in EXPECTED.items():
        path = PACKAGE / 'checkpoints' / filename
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual == expected, filename
        renamed, args = predict_ranges.build_model(str(path), dummy)
        original_constructor = predict_ranges.BPLevelFusionRangePredictor
        try:
            predict_ranges.BPLevelFusionRangePredictor = module.BPLevelFusionRangePredictor
            reference, reference_args = predict_ranges.build_model(str(path), dummy)
        finally:
            predict_ranges.BPLevelFusionRangePredictor = original_constructor
        assert args == reference_args
        renamed.eval()
        reference.eval()
        # Reduced synthetic input: this checks numerical rename compatibility,
        # not a substitute for full-range biological inference.
        renamed.encoder_chunk_bins = reference.encoder_chunk_bins = None
        sequence = torch.randint(0, 5, (1, 4400))
        tracks = torch.randn(1, 4400, 2)
        with torch.inference_mode():
            result = renamed(sequence, tracks)
            old_result = reference(sequence, tracks)
        assert torch.equal(result, old_result), filename
        assert torch.isfinite(result).all()
        checks.append(dict(file=filename, sha256=actual, strict_state_dict_loaded=True,
                           synthetic_forward_equal=True, output_shape=list(result.shape),
                           epoch=int(torch.load(path, map_location='cpu', weights_only=False).get('epoch', -1)),
                           args=args))
        del renamed, reference
        print(f'PASS {filename}: exact hash, strict load, identical synthetic forward', flush=True)

    with tempfile.TemporaryDirectory(prefix='strandabc_validation_') as tmp:
        tmp = Path(tmp)
        records = []
        for start, values in [(0, [1., 9., 2.]), (10, [-2., -3.])]:
            for value in values:
                records.append(dict(chrom='chr8', start=start, end=start+10,
                                    range_start=0, range_end=20,
                                    target_score=value/2, pred_score=value))
        path = tmp / 'predictions.tsv'
        pd.DataFrame(records).to_csv(path, sep='\t', index=False)
        expected = pd.DataFrame(records).groupby(['chrom','start','end']).pred_score.mean().to_numpy()
        for chunksize in [1, 2, 3, 4, 100]:
            result, raw_count, center_count = aggregate.dedupe_source_bins(path, {'chr8'}, 20, chunksize)
            np.testing.assert_allclose(result.pred_score, expected, atol=1e-12)
            np.testing.assert_allclose(result.source_target_score, expected/2, atol=1e-12)
            assert raw_count == center_count == 5
        label = tmp / 'label.bw'
        with pyBigWig.open(str(label), 'w') as bw:
            bw.addHeader([('chr8', 20)])
            bw.addEntries(['chr8']*2, [0, 10], ends=[10, 20], values=[1., -1.])
        bins = aggregate.label_bw_100kb('test', {'test': str(label)}, ['chr8'], 10, 0.)
        np.testing.assert_array_equal(bins.target_score_100kb, [1., -1.])
        np.testing.assert_array_equal(bins.target_label, [1, 0])

    entrypoints = ['train_fusion_compartment.py', 'train_fusion_compartment_range.py',
                  'train_multicell_fusion_compartment_range.py',
                  'scripts/predict_ranges.py', 'scripts/aggregate_predictions.py']
    for entry in entrypoints:
        subprocess.run([sys.executable, str(PACKAGE / entry), '--help'], cwd=PACKAGE,
                       check=True, stdout=subprocess.DEVNULL, timeout=60)
    for shell in (PACKAGE / 'scripts').glob('*.sh'):
        subprocess.run(['bash', '-n', str(shell)], check=True)
    report = dict(passed=True, checkpoints_tested=3, checkpoint_checks=checks,
                  upstream_revision=revision, upstream_url='https://github.com/rrrrraradis/StrandComp',
                  synthetic_cpu_forward_only=True, chunk_size_invariant=True,
                  exact_label_means=True, cli_help_checked=entrypoints,
                  python=sys.executable, torch_version=torch.__version__)
    (destination / 'checkpoints.json').write_text(json.dumps(checks, indent=2) + '\n')
    (destination / 'release_qa.json').write_text(json.dumps(report, indent=2) + '\n')
    print('PASS aggregation chunk sizes, exact labels, all CLI entrypoints and shell syntax', flush=True)


if __name__ == '__main__':
    main()
