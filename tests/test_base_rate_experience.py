"""base_rate_from_experience accepts the canonical Experience."""
import pandas as pd
import pytest
from actuarialpy import Experience

import ratingmodels as rm


def test_experience_and_dataframe_paths_agree():
    df = pd.DataFrame({"mm": [10.0, 12.0, 11.0], "claims": [900.0, 1_150.0, 1_020.0],
                       "rel": [1.0, 1.1, 0.95]})
    via_df = rm.base_rate_from_experience(df, exposure="mm", loss="claims", relativity="rel")
    via_exp = rm.base_rate_from_experience(Experience(df, expense="claims", exposure="mm"), relativity="rel")
    assert via_exp.base_loss_cost == pytest.approx(via_df.base_loss_cost)


def test_dataframe_without_columns_says_how_to_fix():
    with pytest.raises(TypeError, match="pass an Experience"):
        rm.base_rate_from_experience(pd.DataFrame({"a": [1.0]}))
