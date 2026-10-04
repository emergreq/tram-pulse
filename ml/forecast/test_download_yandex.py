"""Network-independent tests: streaming, resume, cache and integrity checks."""
import hashlib
import io
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from ml.forecast import download_yandex as yd


class Response(io.BytesIO):
    def __init__(self, data, status=200, headers=None):
        super().__init__(data)
        self.status = status
        self.headers = headers or {}


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w') as z:
            z.writestr('labels/synthetic.txt', 'artificial fixture')
        self.blob = buffer.getvalue()
        self.meta = {'type': 'file', 'name': 'dataset.zip', 'size': len(self.blob),
                     'sha256': hashlib.sha256(self.blob).hexdigest()}

    def tearDown(self):
        self.tmp.cleanup()

    def api(self, suffix, params):
        if suffix == '/download':
            self.assertEqual(params['path'], '/dataset.zip')
            return {'href': 'https://download.example.invalid/data.zip'}
        return self.meta if 'path' in params else {'type': 'dir'}

    def run_download(self, opener, attempts=2):
        with patch.object(yd, 'api_json', side_effect=self.api), \
             patch.object(yd.urllib.request, 'urlopen', side_effect=opener), \
             patch.object(yd.time, 'sleep'):
            return yd.download(cache_dir=self.root, attempts=attempts)

    def test_folder_download_and_verified_cache(self):
        target = self.run_download(lambda *a, **kw: Response(self.blob))
        self.assertEqual(target.read_bytes(), self.blob)
        with patch.object(yd, 'api_json', side_effect=self.api), \
             patch.object(yd.urllib.request, 'urlopen') as opened:
            self.assertEqual(yd.download(cache_dir=self.root), target)
            opened.assert_not_called()

    def test_interruption_resumes_with_range(self):
        blob = self.blob
        class Interrupted(Response):
            def read(self, size=-1):
                if self.tell():
                    raise OSError('simulated disconnect')
                return super().read(40)
        calls = []
        def open_response(request, **kwargs):
            calls.append(request)
            if len(calls) == 1:
                return Interrupted(blob)
            self.assertEqual(request.get_header('Range'), 'bytes=40-')
            return Response(blob[40:], 206,
                            {'Content-Range': f'bytes 40-{len(blob)-1}/{len(blob)}'})
        self.assertEqual(self.run_download(open_response).read_bytes(), self.blob)
        self.assertEqual(len(calls), 2)

    def test_ignored_range_restarts_instead_of_appending(self):
        class Interrupted(Response):
            def read(self, size=-1):
                if self.tell():
                    raise OSError('disconnect')
                return super().read(40)
        responses = iter([Interrupted(self.blob), Response(self.blob)])
        self.assertEqual(self.run_download(lambda *a, **kw: next(responses)).read_bytes(), self.blob)

    def test_wrong_hash_never_becomes_completed_archive(self):
        self.meta['sha256'] = '0' * 64
        with self.assertRaises(RuntimeError):
            self.run_download(lambda *a, **kw: Response(self.blob), attempts=1)
        self.assertFalse(list(self.root.rglob('dataset.zip')))

    def test_wrong_range_never_becomes_completed_archive(self):
        with self.assertRaises(RuntimeError):
            self.run_download(lambda *a, **kw: Response(self.blob, 206,
                              {'Content-Range': f'bytes 5-{len(self.blob)-1}/{len(self.blob)}'}),
                              attempts=1)
        self.assertFalse(list(self.root.rglob('dataset.zip')))

    def test_api_retries_and_reports_network_problem(self):
        with patch.object(yd.urllib.request, 'urlopen', side_effect=OSError('timeout')) as opened, \
             patch.object(yd.time, 'sleep'):
            with self.assertRaisesRegex(RuntimeError, 'Internet'):
                yd.api_json('', {'public_key': yd.PUBLIC_URL}, attempts=2)
            self.assertEqual(opened.call_count, 2)


if __name__ == '__main__':
    unittest.main()
