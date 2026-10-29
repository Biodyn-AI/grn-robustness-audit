"""The supporting release must detect damage and restore exact original bytes."""
import csv
import gzip
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

MODULE_PATH = Path(__file__).resolve().parents[1] / 'scripts' / 'reproduce_supporting_results.py'
MODULE_SPEC = importlib.util.spec_from_file_location('supporting_results', MODULE_PATH)


class SupportingBundleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = importlib.util.module_from_spec(MODULE_SPEC)
        MODULE_SPEC.loader.exec_module(cls.module)

    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name)
        self.original = b'dataset,value\nkidney,0.25\n'
        self.compressed = gzip.compress(self.original, mtime=0)
        (self.root / 'table.csv.gz').write_bytes(self.compressed)
        self.row = {
            'path': 'table.csv.gz', 'restore_path': 'axis2/table.csv',
            'sha256': hashlib.sha256(self.compressed).hexdigest(),
            'uncompressed_sha256': hashlib.sha256(self.original).hexdigest(),
        }

    def test_round_trip_preserves_bytes(self):
        self.module.verify_file(self.root, self.row)
        target = self.root / 'restored'
        self.module.restore_file(self.root, target, self.row)
        self.assertEqual((target / 'axis2/table.csv').read_bytes(), self.original)

    def test_rejects_corrupted_file(self):
        (self.root / 'table.csv.gz').write_bytes(b'corrupted')
        with self.assertRaises(ValueError):
            self.module.verify_file(self.root, self.row)

    def test_rejects_corrupted_decompressed_hash(self):
        self.row['uncompressed_sha256'] = '0' * 64
        with self.assertRaises(ValueError):
            self.module.verify_file(self.root, self.row)

    def test_rejects_paths_outside_bundle(self):
        for malicious in ['../outside', '/tmp/outside']:
            with self.assertRaises(ValueError):
                self.module.safe_path(self.root, malicious)

    def test_mvcc_uses_means_across_draws(self):
        path = self.root / 'curves.csv'
        path.write_text('dataset,scorer,anchor_cells,cells,k,draw,jaccard_mean\n'
                        'kidney,pearson,3000,100,1000,0,0.8\n'
                        'kidney,pearson,3000,100,1000,1,0.0\n'
                        'kidney,pearson,3000,200,1000,0,0.6\n'
                        'kidney,pearson,3000,200,1000,1,0.6\n')
        estimates = self.module.estimate_mvcc(path)
        self.assertEqual(estimates, [{'dataset': 'kidney', 'scorer': 'pearson',
                                     'anchor_cells': 3000, 'mvcc_cells': 200}])


if __name__ == '__main__':
    unittest.main()
