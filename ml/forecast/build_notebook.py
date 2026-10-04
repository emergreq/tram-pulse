"""Build a standalone Kaggle notebook with automatic Yandex Disk download from the reviewed DS-1 modules."""
import hashlib
import json
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEST = Path(__file__).with_name('DS1_Kaggle_Train.ipynb')
cells = []


def cell(kind, source):
    entry = {'cell_type': kind, 'metadata': {}, 'id': f'ds1-{len(cells):02d}',
             'source': textwrap.dedent(source).strip() + '\n'}
    if kind == 'code':
        entry.update(execution_count=None, outputs=[])
    cells.append(entry)


cell('markdown', '''
# Трамвайный прогноз · DS‑1 · обучение на Kaggle (LightGBM, CatBoost, Календарь РФ 2025)

**Включите Internet и нажмите Run All. GPU не нужен.** Архив автоматически
скачается с https://disk.yandex.ru/d/DiFwlfMOauxjBg. Код моделей уже находится
внутри ноутбука: подключать GitHub или вводить токены не нужно.

1. Создайте Kaggle Notebook и импортируйте этот `.ipynb`.
2. В настройках сессии включите **Internet**. Добавлять данные через Add Input
   не требуется: ноутбук скачает `dataset.zip` по указанной публичной ссылке
   (около 2,54 ГБ), проверит размер/хеш и извлечёт только метки и шаблон.
3. Оставьте **Accelerator: None / CPU** и выполните все ячейки.
4. Скачайте `submission.csv` для платформы и `report.json` для проверки качества.
   Не публикуйте ноутбук с входными данными и обученным артефактом.

Цель — **валидации/час**, а не фактическая заполненность салона.
Модели:
- **LightGBM Direct** (L1 / MAE оптимизация под WAPE числитель)
- **CatBoost Direct** (MAE оптимизация)
- **CalendarProfile** (адаптивный профиль с учетом производственного календаря РФ 2025: рабочая суббота 01.11, нерабочие 03–04.11 и 31.12)
- **Blend Ensemble** (взвешенное ансамблирование)
- Логика возобновления маршрута 50 с 15.11.2025
- Маршрут 5: нулевой cold start на 1464 строки, как требуется шаблоном.
''')

cell('markdown', '## 1. Настройки\nСсылка уже указана. Режим `yandex` скачивает архив автоматически; `input` оставлен для ранее загруженных файлов.')
cell('code', '''
import os
import sys
import json
import csv
import hashlib
import subprocess
import tempfile
import zipfile
import importlib.metadata
from pathlib import Path

PROFILES_ONLY = False  # True: только baseline, без бустинга; другой набор кандидатов.
RUN_TESTS = True
DATA_SOURCE = 'yandex'  # 'input' — необязательный ручной вариант через Add Input.
YANDEX_URL = 'https://disk.yandex.ru/d/DiFwlfMOauxjBg'
DOWNLOAD_CACHE = '/tmp/mthack_yandex'
INPUT_ROOT = Path(os.environ.get('MTHACK_INPUT_DIR', '/kaggle/input'))
WORK = Path(os.environ.get('MTHACK_WORK_DIR', '/kaggle/working'))
INPUT_FILES = {
    'train': '',   # например /kaggle/input/my-private-data/labels_day_train.csv
    'test': '',    # labels_day_test.csv — фактические метки сентября–октября
    'sample': '',  # исходный test_submission.csv за ноябрь–декабрь
}
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
if sys.version_info < (3, 11):
    raise RuntimeError('Нужен Python 3.11+; выберите актуальную Python-среду Kaggle.')
WORK.mkdir(parents=True, exist_ok=True)
RUNTIME = Path(tempfile.mkdtemp(prefix='mthack_ds1_code_'))
EXTRACTED = Path(tempfile.mkdtemp(prefix='mthack_ds1_inputs_'))
OUT = WORK / 'ds1_results'
print('Python:', sys.version.split()[0])
if not PROFILES_ONLY:
    for package in ['numpy', 'scipy', 'scikit-learn', 'lightgbm', 'catboost']:
        try:
            print(package, importlib.metadata.version(package))
        except importlib.metadata.PackageNotFoundError:
            print(package, 'не установлен (будет использован fallback при необходимости)')
print('Результаты:', OUT)
''')

cell('markdown', '''
## 2. Встроенный код модели
Ячейка разворачивает точную копию модулей DS‑1 во временной папке.
Ничего не скачивается. Исходные данные не входят в ноутбук.
''')
paths = ['ml/forecast/core.py', 'ml/forecast/models.py', 'ml/forecast/run.py',
         'ml/forecast/prepare_inputs.py', 'ml/forecast/test_pipeline.py',
         'ml/forecast/download_yandex.py', 'ml/forecast/test_download_yandex.py',
         'ml/submit/adapter.py']
sources = {name: (ROOT / name).read_text() for name in paths}
source_hashes = {name: hashlib.sha256(source.encode()).hexdigest() for name, source in sources.items()}
cell('code', 'SOURCES = ' + repr(sources) + '\nSOURCE_HASHES = ' + repr(source_hashes) + '''
for name, source in SOURCES.items():
    if hashlib.sha256(source.encode()).hexdigest() != SOURCE_HASHES[name]:
        raise RuntimeError('Изменился встроенный исходник: ' + name)
    target = RUNTIME / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source, encoding='utf-8')
print('Развёрнуты и проверены', len(SOURCES), 'модулей DS‑1.')

def run_module(module, *arguments):
    command = [sys.executable, '-u', '-m', module, *map(str, arguments)]
    with subprocess.Popen(command, cwd=RUNTIME, env=os.environ.copy(),
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, bufsize=1) as process:
        for line in process.stdout:
            print(line, end='', flush=True)
        result = process.wait()
    if result:
        raise RuntimeError(f'{module} завершился с ошибкой {result}. См. вывод выше.')

if RUN_TESTS:
    run_module('unittest', 'discover', '-s', 'ml/forecast', '-p', 'test_*.py', '-v')
''')

cell('markdown', '''
## 3. Скачать архив с Яндекс Диска и найти входы
Официальный публичный API не требует токена. Скачивание идёт потоком в `/tmp`,
с прогрессом, повторными попытками, проверкой размера и хеша. При повторном запуске
проверенный архив используется снова; частичная загрузка докачивается, если сервер
поддерживает Range. В память целый архив не загружается.

Распаковываются только небольшие метки и шаблон; сырые транзакции остаются в ZIP.
При сетевой ошибке ячейка остановится с объяснением, а не начнёт обучение без данных.
Официальный способ получения ссылки: https://yandex.cloud/en/docs/datasphere/operations/data/connect-to-ya-disk
''')
cell('code', '''
SEARCH_ROOT = INPUT_ROOT
if DATA_SOURCE == 'yandex':
    import importlib.util
    spec = importlib.util.spec_from_file_location('ds1_yandex', RUNTIME / 'ml/forecast/download_yandex.py')
    downloader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(downloader)
    downloaded_archive = downloader.download(YANDEX_URL, DOWNLOAD_CACHE)
    SEARCH_ROOT = downloaded_archive.parent
elif DATA_SOURCE != 'input':
    raise ValueError("DATA_SOURCE должен быть 'yandex' или 'input'")
if not SEARCH_ROOT.is_dir():
    raise FileNotFoundError('Нет входной папки. Для ручного режима добавьте набор через Add Input.')
required = {'train': 'labels_day_train.csv', 'test': 'labels_day_test.csv',
            'sample': 'test_submission.csv'}
files = {}
for role, name in required.items():
    if INPUT_FILES[role]:
        path = Path(INPUT_FILES[role])
        if not path.is_file():
            raise FileNotFoundError(path)
        files[role] = path
    else:
        matches = sorted(SEARCH_ROOT.rglob(name))
        if len(matches) > 1:
            raise RuntimeError(f'Несколько {name}: {matches}. Укажите INPUT_FILES[{role!r}].')
        if matches:
            files[role] = matches[0]

missing = set(required) - set(files)
archive_candidates = {role: [] for role in missing}
if missing:
    for archive in sorted(SEARCH_ROOT.rglob('*.zip')):
        with zipfile.ZipFile(archive) as bundle:
            for info in bundle.infolist():
                for role in missing:
                    if Path(info.filename).name == required[role]:
                        archive_candidates[role].append((archive, info.filename, info.file_size))
    for role in sorted(missing):
        matches = archive_candidates[role]
        if len(matches) != 1:
            raise RuntimeError(f'{required[role]}: найдено вариантов в ZIP: {len(matches)}. '
                               'Добавьте этот файл отдельно и укажите путь в INPUT_FILES.')
        archive, member, size = matches[0]
        if size > 20_000_000:
            raise ValueError('Неожиданно большой файл меток/шаблона: ' + member)
        with zipfile.ZipFile(archive) as bundle:
            path = EXTRACTED / required[role]
            path.write_bytes(bundle.read(member))
            files[role] = path

for role, path in files.items():
    print(role, '→', path, '|', f'{path.stat().st_size:,}', 'байт')
# Описание внутри исходного ZIP тоже можно прочитать без извлечения raw CSV.
for archive in sorted(SEARCH_ROOT.rglob('*.zip')):
    with zipfile.ZipFile(archive) as bundle:
        descriptions_in_zip = [i for i in bundle.infolist()
                               if Path(i.filename).name.lower() == 'readme.md' and i.file_size < 200_000]
        if len(descriptions_in_zip) == 1:
            description_path = EXTRACTED / 'README.md'
            description_path.write_bytes(bundle.read(descriptions_in_zip[0]))
            print('Описание из архива:\\n', description_path.read_text(encoding='utf-8-sig'))
descriptions = sorted(SEARCH_ROOT.rglob('README.md'))
if len(descriptions) == 1 and descriptions[0].stat().st_size < 200_000:
    print('\\nОписание набора:\\n', descriptions[0].read_text(encoding='utf-8-sig'))
else:
    print('Описание не показано автоматически; сверьте README организаторов отдельно.')
''')

cell('markdown', '''
## 4. Обучение и временная проверка

Все семь кандидатов сравниваются на мае–июне и июле–августе. Затем выбранный
кандидат и два baseline проверяются на сентябре–октябре. Если кандидат хуже
контролей, остаётся baseline. Финальная модель переобучается на январе–октябре.

Логи показывают WAPE-score: **выше лучше**. Результат сентября–октября участвует
в выборе модели и не является независимым тестом. Ноябрь–декабрь прогнозируется
без новых фактов; в рекурсивной модели лаги обновляются предсказаниями.
''')
cell('code', '''
arguments = ['--train', files['train'], '--test', files['test'],
             '--sample', files['sample'], '--out', OUT]
if PROFILES_ONLY:
    arguments.append('--profiles-only')
# Используется только при локальной проверке самого ноутбука; на Kaggle не задавать.
if os.environ.get('MTHACK_NOTEBOOK_TEST') == 'synthetic':
    arguments.append('--synthetic')
run_module('ml.forecast.run', *arguments)
''')

cell('markdown', '## 5. Результаты проверки\nСводные метрики и метрики по маршрутам; score платформы станет известен только после подачи CSV.')
cell('code', '''
from IPython.display import display, Markdown, FileLink
report = json.loads((OUT / 'report.json').read_text())
lines = ['| Период | Модель | WAPE-score | Строк | Пропущено меток |',
         '|---|---|---:|---:|---:|']
for period, models in report['folds'].items():
    for name, result in models.items():
        metric = result['metrics']['all']
        lines.append(f"| {period} | {name} | {metric['wape_score']:.6f} | "
                     f"{metric['rows']} | {metric['missing_label_rows']} |")
display(Markdown('\\n'.join(lines)))
chosen = report['selection']['selected']
print('Выбрана модель:', chosen)
print('Версия:', report['model_version'])
print('Cold start:', report['cold_start'])
lines = ['| Маршрут, сентябрь–октябрь | WAPE-score | Сумма факта | Сумма абсолютных ошибок |',
         '|---|---:|---:|---:|']
for group, metric in report['folds']['sep_oct'][chosen]['metrics'].items():
    if group.startswith('route:'):
        value = metric['wape_score']
        text = f'{value:.6f}' if value is not None else 'не определён'
        lines.append(f"| {group[6:]} | {text} | {metric['actual_sum']} | {metric['absolute_error_sum']} |")
display(Markdown('\\n'.join(lines)))
display(Markdown('**Неизвестный ноябрь–декабрь:** фактическая метрика пока отсутствует. '
                 'Праздники/переносы дней, погода и проверенное расписание пока не включены.'))
''')

cell('markdown', '''
## 6. Скачать результат

Повторно проверяем CSV. Готовые файлы появятся также в **Output**:
- `submission.csv` — файл для конкурсной платформы;
- `report.json` — пришлите его DS‑1 для разбора метрик;
- `forecast_api.csv` — почасовой прогноз для fullstack;
- `ds1_results/model.local.pkl` — локальный обученный артефакт, не публиковать.

Ноутбук сам ничего не отправляет на платформу. Сохраните версию с результатами,
чтобы не потерять файлы после завершения сессии.
''')
cell('code', '''
import shutil
run_module('ml.submit.adapter', '--sample', files['sample'],
           '--submission', OUT / 'submission.csv')
is_synthetic = report['data_kind'] == 'synthetic_not_for_submission'
if is_synthetic:
    print('ВНИМАНИЕ: искусственные данные. Этот CSV НЕ подавать на платформу.')
for name in ['submission.csv', 'report.json', 'forecast_api.csv']:
    target = WORK / name
    shutil.copy2(OUT / name, target)
    print(target.name, '|', f'{target.stat().st_size:,}', 'байт')
    # В Kaggle ссылки ведут на файлы относительно рабочей папки ноутбука.
    display(FileLink(str(target.relative_to(Path.cwd())) if target.is_relative_to(Path.cwd())
                     else str(target)))
with (WORK / 'submission.csv').open(encoding='utf-8', newline='') as f:
    preview = csv.reader(f, delimiter=';')
    for _, row in zip(range(6), preview):
        print(';'.join(row))
print('Готово. Пришлите report.json для оценки реального качества.')
''')

cell('markdown', '''
### Воспроизводимость
В `report.json` сохранены SHA256 входов, исходников и выходов, версии библиотек,
параметры моделей и правила выбора. Пакеты Kaggle не обновляются автоматически:
используется установленная среда, её версии записываются в отчёт.
Исходники: `Dianka678/mthack-tram`, ветка `ds1/forecast-submission`, PR #3.
Официальная справка Kaggle: https://www.kaggle.com/docs/notebooks
''')

notebook = {'cells': cells, 'metadata': {
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.12.0'},
    'kaggle': {'isInternetEnabled': True, 'isGpuEnabled': False,
               'accelerator': 'none', 'dataSources': []},
    'ds1': {'embedded_sha256': source_hashes, 'data_included': False}},
    'nbformat': 4, 'nbformat_minor': 5}
DEST.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
print(DEST)
