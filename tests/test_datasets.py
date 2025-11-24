import pandas as pd

# rtk
from rtk import datasets


class TestDatasets:
    def test_load_mimic_gt_data(self):
        """Test the loading of MIMIC ground truth data."""
        for c in datasets.MIMIC_CLASS_NAMES:
            gt_df = datasets.load_mimic_gt_data(positive_class=c)
            assert isinstance(gt_df, pd.DataFrame)
            assert all(
                col in gt_df.columns for col in ["dicom_id", "image_files", "reports", "y_true"]
            )
            assert gt_df["y_true"].isin([0, 1]).all()
