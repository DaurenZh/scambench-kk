"""Reject wrong rows, wrong label mappings, invalid outputs and stale metrics."""
import copy
import pathlib
import tempfile
import unittest
import zipfile

from import_gpu_results import REQUIRED, align_predictions, source_files, verify_metrics
from train_transformer import score


class ImportTests(unittest.TestCase):
    def setUp(self):
        self.reference=[dict(id=f"case{i}",family=f"family{i}",label=i) for i in range(3)]
        self.rows=[dict(r,predicted=r["label"]) for r in self.reference]

    def test_aligns_by_id_without_relying_on_file_order(self):
        result=align_predictions(list(reversed(self.rows)),self.reference)
        self.assertEqual(result,self.rows)

    def test_rejects_duplicate_ids_and_changed_reference_labels(self):
        rows=copy.deepcopy(self.rows); rows[1]["id"]=rows[0]["id"]
        with self.assertRaises(ValueError): align_predictions(rows,self.reference)
        rows=copy.deepcopy(self.rows); rows[0]["label"]=2
        with self.assertRaises(ValueError): align_predictions(rows,self.reference)

    def test_rejects_out_of_range_prediction(self):
        rows=copy.deepcopy(self.rows); rows[0]["predicted"]=3
        with self.assertRaises(ValueError): align_predictions(rows,self.reference)

    def test_recomputes_scores_and_rejects_validation_metrics_as_test(self):
        values=score([0,1,2],[0,1,2])
        self.assertEqual(verify_metrics(self.rows,values)["macro_f1"],1)
        values["macro_f1"]=.9039
        with self.assertRaisesRegex(ValueError,"disagrees"): verify_metrics(self.rows,values)

    def test_reads_only_the_completed_json_set_from_nested_zip(self):
        with tempfile.TemporaryDirectory() as folder:
            path=pathlib.Path(folder)/"results.zip"
            with zipfile.ZipFile(path,"w") as z:
                for name in REQUIRED: z.writestr("gpu_results/full/"+name,"{}")
                z.writestr("gpu_results/smoke/run.json","{}")
                z.writestr("../not_extracted.txt","must not be written")
            files=source_files(path)
            self.assertEqual(set(files),set(REQUIRED))
            self.assertFalse((path.parent/"not_extracted.txt").exists())


if __name__=="__main__": unittest.main()
