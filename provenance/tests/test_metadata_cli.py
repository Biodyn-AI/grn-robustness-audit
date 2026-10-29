"""Metadata inspection must honor explicit roots and make --help side-effect free."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'inspect_h5ad_metadata.py'
SNAPSHOTS = ('cohort_metadata.csv', 'cohort_category_counts.csv',
             'cohort_overlap.csv', 'h5ad_source_metadata.json')


class MetadataCliTests(unittest.TestCase):
    def test_help_does_not_inspect_or_write(self):
        with tempfile.TemporaryDirectory() as temporary:
            scratch = Path(temporary)
            copied = scratch / 'a/b/c/provenance/inspect_h5ad_metadata.py'
            copied.parent.mkdir(parents=True)
            shutil.copyfile(SCRIPT, copied)
            destination = scratch / 'must_not_exist'
            completed = subprocess.run([sys.executable, str(copied), '--help',
                                        '--out-dir', str(destination)],
                                       cwd=scratch, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn('--workspace', completed.stdout)
            self.assertIn('--out-dir', completed.stdout)
            self.assertFalse(destination.exists())
            self.assertEqual(list(scratch.rglob('*.csv')), [])
            self.assertEqual(list(scratch.rglob('*.json')), [])

    @unittest.skipUnless(os.environ.get('FRAGILITY_METADATA_TEST_WORKSPACE'),
                         'requires the original local H5AD layout')
    def test_explicit_workspace_export_matches_retained_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / 'exported'
            completed = subprocess.run([
                sys.executable, str(SCRIPT), '--workspace',
                os.environ['FRAGILITY_METADATA_TEST_WORKSPACE'],
                '--out-dir', str(destination)], capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            for name in SNAPSHOTS:
                self.assertEqual((destination / name).read_bytes(),
                                 (SCRIPT.parent / name).read_bytes(), name)


if __name__ == '__main__':
    unittest.main()
