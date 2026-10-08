#!/usr/bin/env python
"""Build and verify a portable code + weights bundle without Git metadata."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    report = json.loads((ROOT / 'validation/release_qa.json').read_text())
    assert report['passed']
    for check in report['checkpoint_checks']:
        path = ROOT / 'strandabc/checkpoints' / check['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == check['sha256']
    output = ROOT.parent / 'StrandABC_20261008.zip'
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(ROOT.rglob('*')):
            rel = path.relative_to(ROOT)
            if not path.is_file() or '.git' in rel.parts or '__pycache__' in rel.parts:
                continue
            if path.suffix in {'.pyc', '.log'}:
                continue
            archive.write(path, Path('StrandABC') / rel)
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        for check in report['checkpoint_checks']:
            raw = archive.read(f'StrandABC/strandabc/checkpoints/{check["file"]}')
            assert hashlib.sha256(raw).hexdigest() == check['sha256']
    print(output, flush=True)
    print(f'{output.stat().st_size / 1024**2:.1f} MiB; ZIP integrity and checkpoint hashes verified')


if __name__ == '__main__':
    main()
