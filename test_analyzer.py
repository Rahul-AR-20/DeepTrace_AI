"""
Comprehensive Test Suite for DeepTrace AI:
Validates confidence thresholds, "Unable to Predict" suppression,
input validation, KB fallback, and multi-model consistency.
"""
import sys
import unittest
from predict import (
    ErrorAnalyzer,
    DEFAULT_CONF_THRESHOLD,
    UNABLE_TO_PREDICT_LABEL,
    UNABLE_TO_PREDICT_ALERT,
    INVALID_INPUT_MESSAGE,
)
from knowledge_base import lookup, NO_FIX_MESSAGE, NO_CAUSE_MESSAGE


class TestDeepTraceAI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\nLoading models for test suite...")
        cls.bilstm_analyzer = ErrorAnalyzer(model_file="bilstm.keras")
        cls.lstm_analyzer = ErrorAnalyzer(model_file="lstm.keras")
        cls.cnn_analyzer = ErrorAnalyzer(model_file="cnn.keras")
        print("Models loaded successfully.")

    # -------------------------------------------------------------
    # 1. EMPTY AND INVALID INPUTS
    # -------------------------------------------------------------
    def test_empty_and_whitespace_inputs(self):
        cases = ["", "   ", "\t\n", None]
        for c in cases:
            res = self.bilstm_analyzer.analyze(c)
            self.assertIn("error", res, f"Expected error for input {repr(c)}")
            self.assertEqual(res["error"], INVALID_INPUT_MESSAGE)

    def test_non_meaningful_inputs(self):
        cases = [
            "....",
            "!@#$%^&*()_+",
            "123 4567 890",
            "the a an of to in",
        ]
        for c in cases:
            res = self.bilstm_analyzer.analyze(c)
            self.assertIn("error", res, f"Expected error for non-meaningful input {repr(c)}")
            self.assertEqual(res["error"], INVALID_INPUT_MESSAGE)

    # -------------------------------------------------------------
    # 2. ALL SEVEN CATEGORIES (KNOWN ERRORS)
    # -------------------------------------------------------------
    def test_all_seven_categories(self):
        known_samples = {
            "Syntax": "SyntaxError: invalid syntax",
            "Runtime": "ZeroDivisionError: division by zero",
            "Type": "TypeError: unsupported operand type(s) for +: 'int' and 'str'",
            "Dependency": "ModuleNotFoundError: No module named 'requests'",
            "Database": "psycopg2.OperationalError: could not connect to server: Connection refused",
            "Network": "requests.exceptions.ConnectionError: HTTPSConnectionPool(host='api.example.com', port=443): Max retries exceeded",
            "API": "requests.exceptions.HTTPError: 401 Client Error: Unauthorized for url: https://api.openai.com/v1",
        }

        for expected_cat, error_msg in known_samples.items():
            res = self.bilstm_analyzer.analyze(error_msg, conf_threshold=0.60)
            self.assertNotIn("error", res)
            self.assertFalse(res["uncertain"], f"Sample for {expected_cat} unexpectedly flagged uncertain: {res}")
            self.assertEqual(res["error_type"], expected_cat)
            self.assertGreaterEqual(res["confidence"], 0.60)

    # -------------------------------------------------------------
    # 3. UNFAMILIAR AND IRRELEVANT TEXT (UNABLE TO PREDICT)
    # -------------------------------------------------------------
    def test_unfamiliar_and_irrelevant_text(self):
        irrelevant_cases = [
            "The quick brown fox jumps over the lazy dog in the sunny park.",
            "I went to the supermarket and bought groceries yesterday.",
            "Can you please explain quantum mechanics to me?",
        ]
        for text in irrelevant_cases:
            res = self.bilstm_analyzer.analyze(text, conf_threshold=0.70)
            self.assertNotIn("error", res)
            self.assertTrue(res["uncertain"], f"Text should be uncertain: {text}")
            self.assertEqual(res["error_type"], UNABLE_TO_PREDICT_LABEL)
            self.assertEqual(res["status_message"], UNABLE_TO_PREDICT_ALERT)
            self.assertLess(res["confidence"], 0.70)
            # Verify raw prediction is preserved separately for debugging
            self.assertIn(res["raw_error_type"], ["Syntax", "Runtime", "Type", "Dependency", "Database", "Network", "API"])

    # -------------------------------------------------------------
    # 4. CONFIGURABLE THRESHOLD & SUPPRESSION BEHAVIOR
    # -------------------------------------------------------------
    def test_configurable_threshold_suppression(self):
        msg = "Some borderline exception message in python runtime environment"
        res_baseline = self.bilstm_analyzer.analyze(msg, conf_threshold=0.10)
        baseline_conf = res_baseline["confidence"]

        # If threshold is above confidence, it MUST display "Unable to Predict"
        res_high_thresh = self.bilstm_analyzer.analyze(msg, conf_threshold=0.99)
        self.assertTrue(res_high_thresh["uncertain"])
        self.assertEqual(res_high_thresh["error_type"], UNABLE_TO_PREDICT_LABEL)
        self.assertEqual(res_high_thresh["status_message"], UNABLE_TO_PREDICT_ALERT)
        self.assertEqual(res_high_thresh["raw_error_type"], res_baseline["raw_error_type"])

    # -------------------------------------------------------------
    # 5. KNOWLEDGE BASE FALLBACK (NO SPECIFIC FIX FOUND)
    # -------------------------------------------------------------
    def test_missing_knowledge_base_rule(self):
        unmatched_error = "CustomApplicationException: internal worker thread timeout in job 42"
        cause, fix, matched = lookup(unmatched_error, "Runtime")
        self.assertFalse(matched)
        self.assertEqual(fix, NO_FIX_MESSAGE)
        self.assertEqual(cause, NO_CAUSE_MESSAGE)

    def test_matched_knowledge_base_rule(self):
        matched_error = "psycopg2.OperationalError: could not connect to server: Connection refused"
        cause, fix, matched = lookup(matched_error, "Database")
        self.assertTrue(matched)
        self.assertNotEqual(fix, NO_FIX_MESSAGE)
        self.assertIn("database", fix.lower())

    # -------------------------------------------------------------
    # 6. LONG TRACEBACK HANDLING
    # -------------------------------------------------------------
    def test_long_traceback(self):
        traceback_text = """
Traceback (most recent call last):
  File "/var/app/services/auth_service.py", line 142, in authenticate_user
    user = db.session.query(User).filter_by(username=username).first()
  File "/var/app/venv/lib/python3.11/site-packages/sqlalchemy/orm/query.py", line 2814, in first
    return self.limit(1).all()[0]
  File "/var/app/venv/lib/python3.11/site-packages/sqlalchemy/orm/query.py", line 2692, in all
    return list(self)
psycopg2.OperationalError: could not connect to server: Connection refused (0x0000274D/10061)
        """
        res = self.bilstm_analyzer.analyze(traceback_text, conf_threshold=0.70)
        self.assertNotIn("error", res)
        self.assertEqual(res["error_type"], "Database")
        self.assertTrue(res["kb_match"])
        self.assertFalse(res["uncertain"])

    # -------------------------------------------------------------
    # 7. MULTI-MODEL COMPARISON (CNN, LSTM, BI-LSTM)
    # -------------------------------------------------------------
    def test_all_models_support_threshold_and_unable_to_predict(self):
        models = [
            ("Bi-LSTM", self.bilstm_analyzer),
            ("LSTM", self.lstm_analyzer),
            ("CNN", self.cnn_analyzer),
        ]
        test_msg = "Random unfamiliar text that should trigger low confidence"
        for name, analyzer in models:
            res = analyzer.analyze(test_msg, conf_threshold=0.70)
            self.assertNotIn("error", res, f"{name} failed to process")
            self.assertTrue(res["uncertain"], f"{name} should flag uncertainty")
            self.assertEqual(res["error_type"], UNABLE_TO_PREDICT_LABEL, f"{name} should display 'Unable to Predict'")
            self.assertIn(res["raw_error_type"], ["Syntax", "Runtime", "Type", "Dependency", "Database", "Network", "API"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
