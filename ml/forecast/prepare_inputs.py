"""Extract only labels/sample/description from a local organizer ZIP, never raw events."""
import argparse
import json
import zipfile
from pathlib import Path

from .core import sha256


def extract(archive, output):
    required = {'labels_day_train.csv', 'labels_day_test.csv', 'test_submission.csv'}
    output = Path(output)
    with zipfile.ZipFile(archive) as z:
        members = {}
        for info in z.infolist():
            name = Path(info.filename).name
            if name not in required and name.lower() != 'readme.md':
                continue
            if name in members:
                raise ValueError(f'Ambiguous archive: duplicate {name}')
            if info.file_size > 20_000_000:
                raise ValueError(f'Unexpectedly large label/description file: {name}')
            members[name] = info
        if not required <= members.keys():
            raise ValueError(f'Missing archive members: {sorted(required - members.keys())}')
        if any((output / name).exists() for name in members):
            raise ValueError('Refuse to overwrite original inputs; use an empty directory')
        output.mkdir(parents=True, exist_ok=True)
        for name, info in members.items():
            (output / name).write_bytes(z.read(info))
    manifest = {'archive_name': Path(archive).name,
                'source_url': 'https://disk.yandex.ru/d/DiFwlfMOauxjBg',
                'files_sha256': {name: sha256(output / name) for name in sorted(members)}}
    (output / 'inputs_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    a = p.parse_args()
    print(json.dumps(extract(a.archive, a.out), indent=2))
