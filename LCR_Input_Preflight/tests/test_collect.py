import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from collect_inputs import collect, verify_assets, read_checked

class CollectorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.assets = self.root / 'assets'
        self.assets.mkdir()
        self.data = b'{"fixture": true}\n'
        (self.assets / 'fixture.json').write_bytes(self.data)
        self.expected = {'fixture.json': hashlib.sha256(self.data).hexdigest()}
        self.bank = self.root / 'permitted_prompts.jsonl'
        self.bank.write_text('{"question":"synthetic fixture"}\n', encoding='utf-8')
    def tearDown(self):
        self.tmp.cleanup()
    def test_verified_assets(self):
        self.assertEqual(verify_assets(self.assets, self.expected)['fixture.json'], self.data)
    def test_hash_mismatch(self):
        with self.assertRaises(ValueError):
            verify_assets(self.assets, {'fixture.json': '0'*64})
    def test_missing_asset(self):
        with self.assertRaises(ValueError):
            verify_assets(self.assets, {'absent.json': '0'*64})
    def test_reject_path_traversal(self):
        with self.assertRaises(ValueError):
            verify_assets(self.assets, {'../fixture.json': '0'*64})
    def test_symlink(self):
        link=self.root/'link.json'; link.symlink_to(self.assets/'fixture.json')
        with self.assertRaises(ValueError): read_checked(link)
    def test_pointer(self):
        p=self.root/'pointer.json'; p.write_bytes(b'version https://git-lfs.github.com/spec/v1\n')
        with self.assertRaises(ValueError): read_checked(p)
    def test_limit(self):
        with self.assertRaises(ValueError): read_checked(self.bank, limit=1)
    def test_exclusion_required(self):
        with self.assertRaises(ValueError):
            collect(self.assets, [], self.root/'out.zip', self.expected)
    def test_weights_rejected(self):
        p=self.root/'model.safetensors'; p.write_bytes(b'not a model')
        with self.assertRaises(ValueError):
            collect(self.assets, [p], self.root/'out.zip', self.expected)
    def test_archive_receipt(self):
        out=self.root/'out.zip'
        manifest=collect(self.assets, [self.bank], out, self.expected)
        with zipfile.ZipFile(out) as z:
            self.assertIsNone(z.testzip())
            for name, entry in manifest['files'].items():
                self.assertEqual(hashlib.sha256(z.read(name)).hexdigest(), entry['sha256'])
            receipt=json.loads(z.read('INPUT_MANIFEST.json'))
            self.assertEqual(receipt['native_tokenization'], 'NOT_RUN')
            self.assertEqual(receipt['model_calls'],0)
    def test_no_overwrite(self):
        out=self.root/'out.zip'; out.write_bytes(b'protected')
        with self.assertRaises(ValueError): collect(self.assets,[self.bank],out,self.expected)
        self.assertEqual(out.read_bytes(),b'protected')
    def test_sources_unchanged(self):
        before=self.bank.read_bytes()
        collect(self.assets,[self.bank],self.root/'out.zip',self.expected)
        self.assertEqual(self.bank.read_bytes(),before)
    def test_duplicate_basenames_safe(self):
        other=self.root/'other'; other.mkdir()
        copy=other/self.bank.name; copy.write_bytes(b'another permitted source')
        out=self.root/'out.zip'
        manifest=collect(self.assets,[self.bank,copy],out,self.expected)
        self.assertEqual(len(manifest['selected_exclusion_inputs']),2)
        with zipfile.ZipFile(out) as z:
            self.assertNotEqual(z.read('exclusion_inputs/01_'+self.bank.name),z.read('exclusion_inputs/02_'+self.bank.name))
    def test_deterministic_archive(self):
        a=self.root/'a.zip'; b=self.root/'b.zip'
        collect(self.assets,[self.bank],a,self.expected)
        collect(self.assets,[self.bank],b,self.expected)
        self.assertEqual(a.read_bytes(),b.read_bytes())

if __name__ == '__main__': unittest.main()
