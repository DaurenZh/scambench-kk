"""Verify family isolation, paired evaluation, and head+tail token handling."""
import copy
import json
import unittest

import pandas as pd

from prepare_gpu_bundle import check_sets, make_sets
from train_transformer import EncodedRows, score


def fixture():
    rows, silver = [], []
    classes = ["LEGITIMATE", "SUSPICIOUS", "SCAM"]
    for i in range(300):
        source = dict(id=f"base::{i:016x}::0", language="en", split="train",
                      label_3class=classes[i % 3], decision_class=["engage_legitimate", "audit", "refuse"][i % 3],
                      messages=json.dumps([{"content": "hello"}]),
                      translated_from=None)
        rows += [source, dict(source, id=source["id"] + "::kk", language="kk",
                              translated_from=source["id"])]
        rows.append(dict(source, id=f"aug::{i:016x}::restyle"))
    for i in range(1000, 1090):
        source = dict(id=f"base::{i:016x}::0", language="en", split="train",
                      label_3class=classes[i % 3], decision_class=["engage_legitimate", "audit", "refuse"][i % 3],
                      messages=json.dumps([{"content": "train"}]),
                      translated_from=None)
        rows += [source, dict(source, id=f"aug::{i:016x}::restyle")]
        silver.append(dict(source, id=source["id"] + "::kk_mt", language="kk",
                           translated_from=source["id"]))
    return pd.DataFrame(rows), pd.DataFrame(silver)


class BundleTests(unittest.TestCase):
    def test_holdout_restylings_and_silver_stay_isolated(self):
        sets = make_sets(*fixture())
        self.assertEqual(len(sets["test_en"]), 300)
        self.assertEqual(len(sets["train_en"]) + len(sets["dev_en"]), 180)
        self.assertEqual(len(sets["train_silver"]) + len(sets["dev_silver"]), 90)
        self.assertTrue(sets["dev_silver"])
        check_sets(sets)
        contaminated = copy.deepcopy(sets)
        contaminated["train_silver"].append(sets["test_kk"][0])
        with self.assertRaisesRegex(ValueError, "crosses"):
            check_sets(contaminated)

    def test_contaminated_silver_is_rejected(self):
        merged, silver = fixture()
        silver.loc[0, "translated_from"] = "base::0000000000000000::0"
        with self.assertRaisesRegex(ValueError, "outside"):
            make_sets(merged, silver)

    def test_mismatched_pairs_are_rejected(self):
        sets = make_sets(*fixture())
        sets["test_kk"][0], sets["test_kk"][1] = sets["test_kk"][1], sets["test_kk"][0]
        with self.assertRaisesRegex(ValueError, "aligned"):
            check_sets(sets)

    def test_silver_action_target_must_match_source(self):
        merged, silver = fixture()
        silver.loc[0, "decision_class"] = "made_up_action"
        with self.assertRaisesRegex(ValueError, "actions differ"):
            make_sets(merged, silver)

    def test_head_and_tail_survive_with_special_tokens(self):
        class Tokenizer:
            def __call__(self, texts, **kwargs):
                return {"input_ids": [[int(x) for x in t.split()] for t in texts]}
            def num_special_tokens_to_add(self, pair):
                return 2
            def build_inputs_with_special_tokens(self, tokens):
                return [100] + tokens + [101]
        encoded = EncodedRows([{"text": "0 1 2 3 4 5 6 7 8 9", "label": 2}], Tokenizer(), 8)
        self.assertEqual(encoded[0]["input_ids"], [100, 0, 1, 2, 7, 8, 9, 101])
        self.assertEqual(encoded.truncated, 1)

    def test_binary_merges_predictions_without_changing_three_class_target(self):
        metrics = score([0, 1, 2], [0, 2, 1])
        self.assertEqual(metrics["binary_macro_f1"], 1.0)
        self.assertLess(metrics["macro_f1"], 0.34)


if __name__ == "__main__":
    unittest.main()
