"""Fresh, isolated dependency installation and steel replay.

Run this utility with any Python >=3.10. Supply an actual Python 3.13
interpreter via --python-executable. The supplied runner is never patched during
execution. Use the current runner and requirements specified in REPLAY.md;
historical stage51 evidence remains a record of the original environment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python-executable', required=True)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--mode', choices=['representative', 'full'], default='representative')
    args = parser.parse_args()
    inp = args.input_dir.resolve()
    out = args.output_dir.resolve()
    if out.exists():
        raise FileExistsError('Use a new output directory; existing evidence is not overwritten.')
    out.mkdir(parents=True)
    start = time.time()
    status = {'stage': 55, 'status': 'STARTED',
              'fresh_environment_replay_completed': False, 'mode': args.mode,
              'source_training_rerun': False,
              'runner_modified_during_bootstrap': False}
    env = dict(os.environ)
    env.update(PYTHONNOUSERSITE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')

    def run(cmd, filename, timeout=1200):
        with (out / filename).open('w', encoding='utf-8') as log:
            result = subprocess.run([str(c) for c in cmd], stdout=log, stderr=subprocess.STDOUT,
                                    env=env, timeout=timeout, check=False)
        if result.returncode:
            raise RuntimeError(f'{filename}: exit {result.returncode}')

    try:
        runner = inp / 'v49_replay.py'
        requirements = inp / 'v49_requirements.txt'
        for p in [runner, requirements]:
            if not p.is_file():
                raise FileNotFoundError(p)
        status['runner_sha256'] = hashlib.sha256(runner.read_bytes()).hexdigest()
        status['requirements_sha256'] = hashlib.sha256(requirements.read_bytes()).hexdigest()
        status['archived_runner_modified'] = status['runner_sha256'] != 'e68e694b583c7d386133d82cb289cdcda09e83a9e419a54ffd4f03a820701bd4'
        run([args.python_executable, '-c',
             'import sys; print(sys.version); assert sys.version_info[:2] == (3,13)'], 'python_version.log')
        run([args.python_executable, '-m', 'venv', str(out / 'venv')], 'venv_creation.log')
        py = out / 'venv' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
        probe = '''import importlib.util,json,sys,site
names=['numpy','scipy','sklearn','lightgbm','rdkit','joblib','threadpoolctl','PIL']
missing={n:importlib.util.find_spec(n) is None for n in names}
result={'python':sys.version,'prefix':sys.prefix,'base_prefix':sys.base_prefix,'site_user_enabled':site.ENABLE_USER_SITE,'absent_before_install':missing}
print(json.dumps(result,indent=2)); assert all(missing.values()); assert sys.prefix != sys.base_prefix; assert not site.ENABLE_USER_SITE
'''
        run([py, '-c', probe], 'before_install.json')
        run([py, '-m', 'pip', 'install', '--disable-pip-version-check', '--no-input',
             '--report', out / 'pip_install_report.json', '-r', requirements], 'pip_install.log')
        run([py, '-m', 'pip', 'check'], 'pip_check.log')
        run([py, '-m', 'pip', 'freeze', '--all'], 'pip_freeze.txt')
        run([py, runner, '--input-dir', inp, '--output-dir', out / 'replay',
             '--mode', args.mode, '--require-isolated'], 'replay.log')
        audit = json.loads((out / 'replay/audit.json').read_text())
        preflight = json.loads((out / 'replay/preflight.json').read_text())
        before = json.loads((out / 'before_install.json').read_text())
        assert audit['status'] == 'PASS' and audit['input_files_unchanged']
        assert preflight['ready'] and preflight['venv_isolated_from_system_packages']
        assert preflight['dependencies_resolved_within_prefix']
        assert all(before['absent_before_install'].values())
        assert hashlib.sha256(runner.read_bytes()).hexdigest() == status['runner_sha256']
        status.update(status='PASS', fresh_environment_replay_completed=True,
                      freshly_installed_packages=before['absent_before_install'],
                      python=preflight['python'],
                      verification=audit,
                      note='This wrapper attests the fresh installation; the supplied runner retains its own false attestation flag. archived_runner_modified compares its hash with the original v49 runner, not with the file at the start of this invocation.')
    except Exception as exc:
        status.update(status='BLOCKED_OR_FAILED', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        status['elapsed_seconds'] = time.time() - start
        (out / 'bootstrap_status.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
        print(json.dumps(status, indent=2), flush=True)


if __name__ == '__main__':
    main()
