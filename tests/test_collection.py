import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.collect_av import main


class CollectionTests(unittest.TestCase):
    def run_collection(self, root, worker):
        args = ['collect_av.py', '--task', 'adjust_bottle', '--episodes', '1',
                '--max-attempts', '2', '--output', root, '--min-free-gib', '0']
        with patch('sys.argv', args), patch('scripts.collect_av.subprocess.run', side_effect=worker), \
             patch('scripts.collect_av.audit_episode', return_value={'all_checks_pass': True, 'success': True}):
            return main()

    def test_failures_are_retained_and_resume_does_not_duplicate(self):
        with tempfile.TemporaryDirectory() as root:
            calls = []
            def worker(cmd, **kwargs):
                seed = int(cmd[cmd.index('--seed') + 1])
                calls.append(seed)
                if seed == 1000:
                    return SimpleNamespace(returncode=2)
                out = Path(root) / 'train/adjust_bottle/phase' / f'seed_{seed:06d}'
                out.mkdir(parents=True)
                (out / 'result.json').write_text(json.dumps({'success': True, 'seed': seed}))
                return SimpleNamespace(returncode=0)
            self.assertEqual(self.run_collection(root, worker), 0)
            self.assertEqual(calls, [1000, 1001])
            self.assertEqual(self.run_collection(root, worker), 0)
            self.assertEqual(calls, [1000, 1001])
            result = json.loads((Path(root) / 'train/adjust_bottle/phase/manifest.json').read_text())
            self.assertEqual(len(result['episodes']), 1)
            self.assertEqual(len(result['attempts']), 2)

    def test_timeout_has_a_failure_receipt(self):
        import subprocess
        with tempfile.TemporaryDirectory() as root:
            def worker(cmd, **kwargs):
                raise subprocess.TimeoutExpired(cmd, 1)
            self.assertEqual(self.run_collection(root, worker), 2)
            receipt = Path(root) / 'train/adjust_bottle/phase/seed_001000/result.json'
            self.assertEqual(json.loads(receipt.read_text())['reason'], 'timeout')

    def test_stale_lock_is_not_silently_removed(self):
        with tempfile.TemporaryDirectory() as root:
            out = Path(root) / 'train/adjust_bottle/phase'
            out.mkdir(parents=True)
            lock = out / '.collection.lock'
            lock.write_text('123')
            with self.assertRaises(FileExistsError):
                self.run_collection(root, lambda *args: None)
            self.assertTrue(lock.exists())


if __name__ == '__main__':
    unittest.main()
