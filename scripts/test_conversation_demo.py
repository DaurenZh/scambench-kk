"""Verify causal prefix replay and absence of label/role metadata in input text."""
import unittest

from conversation_demo import ConversationPredictor
from portable_text_model import flatten
from prepare_conversation_demo import authored_cases


class ConversationTests(unittest.TestCase):
    def test_prefix_replay_does_not_read_future_turns(self):
        class Model:
            def __init__(self): self.seen=[]
            def predict(self,texts):
                self.seen=list(texts)
                return ["SCAM" if "secret" in t else "LEGITIMATE" for t in texts]
        predictor=ConversationPredictor.__new__(ConversationPredictor)
        predictor.risk=Model(); predictor.action=Model()
        messages=[dict(role="user",content="Hello"),dict(role="user",content="Send the secret")]
        result=predictor.replay(messages)
        self.assertEqual(predictor.risk.seen,["Hello","Hello\nSend the secret"])
        self.assertEqual(result[0]["risk"],"LEGITIMATE")
        self.assertEqual(result[1]["risk"],"SCAM")

    def test_reference_and_role_fields_never_enter_flattened_input(self):
        message=dict(role="assistant",content="Hello",expected_risk="SCAM",decision="refuse")
        self.assertEqual(flatten([message]),"Hello")
        with self.assertRaises(ValueError): flatten([])
        with self.assertRaises(ValueError): flatten([dict(content=42)])

    def test_same_intent_variants_preserve_authored_expectations(self):
        cases=authored_cases()["cases"]
        self.assertEqual(len(cases),27)
        lookup={r["id"]:r for r in cases}
        self.assertEqual(len(lookup),27)
        for case in cases:
            if case.get("same_intent_as"):
                original=lookup[case["same_intent_as"]]
                self.assertEqual(case["expected_risk"],original["expected_risk"])
                self.assertEqual(case["acceptable_actions"],original["acceptable_actions"])
                self.assertNotEqual(flatten(case["messages"]),flatten(original["messages"]))


if __name__=="__main__": unittest.main()
