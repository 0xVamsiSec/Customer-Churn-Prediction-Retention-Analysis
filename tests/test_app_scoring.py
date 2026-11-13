import numpy as np
import pandas as pd
import pytest

from app.server import batch, score, settings
from src import preprocessing as prep, risk_scoring as risk


def test_unlabeled_raw_profiles_reproduce_frozen_scores():
    raw = pd.read_csv(prep.RAW_PATH).drop(columns=[prep.TARGET])
    saved = pd.read_csv(risk.SCORES_PATH)

    scored = score(raw)

    pd.testing.assert_series_equal(scored["customerID"], saved["customerID"])
    np.testing.assert_allclose(scored["risk_score"], saved["churn_probability"], rtol=0, atol=1e-12)
    assert scored["risk_segment"].equals(saved["risk_segment"])
    assert scored["flag_at_0_40"].equals(saved["predicted_churn_at_0.40"])


def test_batch_example_matches_single_profile():
    profile = pd.DataFrame([settings()["example"]])
    pd.testing.assert_frame_equal(batch(profile.to_csv(index=False).encode("utf-8")), score(profile))


def test_batch_rejects_csv_without_predictors():
    with pytest.raises(ValueError, match="Missing fields:"):
        batch(b"tenure\n3\n")
