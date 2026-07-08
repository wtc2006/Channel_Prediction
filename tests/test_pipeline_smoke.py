import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from data_generator import generate_channel_data
from dl_model_training import train_lstm
from early_warning import classify_prediction, run_warning_engine
from model_training import create_window_features, train_random_forest


class PipelineSmokeTest(unittest.TestCase):
    def test_window_feature_shape(self):
        data = np.arange(20, dtype=float)

        X, y = create_window_features(data, window_size=5)

        self.assertEqual(X.shape, (15, 6))
        self.assertEqual(y.shape, (15,))

    def test_warning_classification(self):
        self.assertEqual(classify_prediction(14.9), "critical")
        self.assertEqual(classify_prediction(16.5), "warning")
        self.assertEqual(classify_prediction(18.0), "normal")

    def test_small_pipeline_without_plots(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            data_path = temp_path / "channel_data.csv"
            rf_model_path = temp_path / "channel_model.pkl"
            lstm_model_path = temp_path / "channel_lstm_model.pth"

            generate_channel_data(data_size=80, output_path=data_path, show_plot=False)
            rf_result = train_random_forest(
                data_path=data_path,
                model_path=rf_model_path,
                window_size=8,
                show_plot=False,
            )
            lstm_result = train_lstm(
                data_path=data_path,
                model_path=lstm_model_path,
                window_size=8,
                epochs=2,
                show_plot=False,
            )
            warning_result = run_warning_engine(
                data_path=data_path,
                model_path=rf_model_path,
                start_point=20,
                end_point=35,
                window_size=8,
                show_plot=False,
            )

            self.assertTrue(rf_model_path.exists())
            self.assertTrue(lstm_model_path.exists())
            self.assertGreaterEqual(rf_result["mae"], 0)
            self.assertGreaterEqual(lstm_result["mae"], 0)
            self.assertEqual(len(warning_result["predictions"]), 15)


if __name__ == "__main__":
    unittest.main()
