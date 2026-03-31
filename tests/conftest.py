"""
pytest configuration and shared fixtures.
"""
import pytest


@pytest.fixture
def sample_county_fips() -> list[str]:
    return ["06037", "17031", "48113", "04013", "06073"]


@pytest.fixture
def sample_exposure_df(sample_county_fips):
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(42)
    return pd.DataFrame({"county_fips": sample_county_fips, "paraquat_kg": rng.uniform(0, 1000, 5)}).set_index("county_fips")


@pytest.fixture
def sample_outcome_df(sample_county_fips):
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(99)
    return pd.DataFrame({"county_fips": sample_county_fips, "parkinsons_mortality_rate": rng.uniform(1.0, 10.0, 5)}).set_index("county_fips")
