"""Download organizer ZIP through the public Yandex Disk API, without credentials."""
import hashlib
import json
import re
import shutil
import time
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

PUBLIC_URL = 'https://disk.yandex.ru/d/DiFwlfMOauxjBg'
API = 'https://cloud-api.yandex.net/v1/disk/public/resources'


def api_json(suffix, params, attempts=3):
    url = API + suffix + '?' + urllib.parse.urlencode(params)
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                return json.load(response)
        except (OSError, ValueError) as exc:
            if attempt == attempts - 1:
                raise RuntimeError('API Яндекс Диска недоступен. Включите Internet в Kaggle; '
                                   'если уже включён, повторите ячейку позже. '
                                   'Также возможен лимит скачивания публичной ссылки.') from exc
            print(f'API: повторная попытка {attempt + 2}/{attempts}', flush=True)
            time.sleep(2 ** (attempt + 1))


def digest(path, algorithm='sha256'):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, algorithm).hexdigest()


def download(public_url=PUBLIC_URL, cache_dir='/tmp/mthack_yandex', attempts=3):
    params = {'public_key': public_url}
    meta = api_json('', params)
    if meta.get('type') == 'dir':
        # Organizer link is a public folder containing dataset.zip.
        params['path'] = '/dataset.zip'
        meta = api_json('', params)
    if meta.get('type') != 'file' or not meta.get('name', '').lower().endswith('.zip'):
        raise ValueError('Ссылка должна вести на ZIP или папку с dataset.zip')
    expected_size = int(meta['size'])
    if expected_size <= 0:
        raise ValueError('Пустой архив в метаданных Яндекс Диска')
    algorithm = 'sha256' if meta.get('sha256') else 'md5'
    expected_hash = meta.get(algorithm)
    identity = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:16]
    directory = Path(cache_dir) / identity
    directory.mkdir(parents=True, exist_ok=True)
    target, partial = directory / 'dataset.zip', directory / 'dataset.zip.part'

    def valid(path):
        return (path.exists() and path.stat().st_size == expected_size
                and (not expected_hash or digest(path, algorithm) == expected_hash)
                and zipfile.is_zipfile(path))

    if valid(target):
        print('Используется ранее проверенный архив:', target, flush=True)
        return target
    print(f'Архив: {expected_size / 1024**3:.2f} ГиБ. Потоковое скачивание во временную папку.',
          flush=True)
    for attempt in range(attempts):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            if offset >= expected_size:
                if valid(partial):
                    break
                partial.unlink()
                offset = 0
            if shutil.disk_usage(directory).free < expected_size - offset + 64 * 1024**2:
                raise RuntimeError('Недостаточно места для архива во временной папке')
            href = api_json('/download', params)['href']
            if urllib.parse.urlparse(href).scheme != 'https':
                raise ValueError('API вернул незащищённую ссылку скачивания')
            headers = {'Range': f'bytes={offset}-'} if offset else {}
            request = urllib.request.Request(href, headers=headers)
            with urllib.request.urlopen(request, timeout=60) as response:
                status = response.status
                if status == 206:
                    match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',
                                         response.headers.get('Content-Range', ''))
                    if not match or int(match[1]) != offset or int(match[3]) != expected_size:
                        raise ValueError('Некорректный диапазон при докачке')
                    mode = 'ab'
                elif status == 200:
                    offset, mode = 0, 'wb'  # Server ignored Range: safely restart.
                else:
                    raise ValueError(f'Неожиданный HTTP-статус: {status}')
                received, logged = offset, offset
                with partial.open(mode) as f:
                    while chunk := response.read(4 * 1024**2):
                        f.write(chunk)
                        received += len(chunk)
                        if received > expected_size:
                            raise ValueError('Размер загрузки превысил метаданные')
                        if received - logged >= 64 * 1024**2 or received == expected_size:
                            print(f'Скачано {received / expected_size:.1%} '
                                  f'({received / 1024**2:.0f}/{expected_size / 1024**2:.0f} МиБ)',
                                  flush=True)
                            logged = received
            if partial.stat().st_size != expected_size:
                raise OSError('Соединение прервано до конца файла')
            if not valid(partial):
                partial.unlink()
                raise ValueError('Проверка хеша или структуры ZIP не пройдена')
            break
        except (OSError, ValueError) as exc:
            if attempt == attempts - 1:
                raise RuntimeError('Архив не скачан после повторных попыток. '
                                   'Проверьте Internet и повторите ячейку: частичный файл сохранён '
                                   'для докачки, если сервер поддерживает Range.') from exc
            print(f'Скачивание: повторная попытка {attempt + 2}/{attempts}', flush=True)
            time.sleep(2 ** (attempt + 1))
    partial.replace(target)
    manifest = {'source_url': public_url, 'path': params.get('path'), 'size': expected_size,
                'sha256': digest(target), 'fetched_at_utc': datetime.now(timezone.utc).isoformat(),
                'server_hash_algorithm': algorithm if expected_hash else None,
                'server_hash_verified': bool(expected_hash)}
    (directory / 'download_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('Архив скачан и проверен:', target, flush=True)
    return target
