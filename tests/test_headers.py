import numpy as np
import pandas as pd

from src.constants import ID_COL
from src.data.headers import annotate, laterality


def _meta(**cols) -> pd.DataFrame:
    n = len(next(iter(cols.values())))
    base = {ID_COL: [f"s{i}" for i in range(n)], "SeriesInstanceUID": [f"x{i}" for i in range(n)]}
    return pd.DataFrame({**base, **cols})


def test_fat_suppression_from_description_and_scan_options() -> None:
    df = annotate(
        _meta(
            SeriesDescription=["DP SPAIR_CS", "Water: SMART FAT", "T1_TSE_CS", "PD_TSE", "pd_tse"],
            ScanOptions=[None, None, None, None, "PFP|FS"],
        )
    )
    assert df["fatsat"].tolist() == [True, True, False, False, True]


def test_weighting_from_description_then_tr_te() -> None:
    df = annotate(
        _meta(
            SeriesDescription=[
                "t1_tse_cor",
                "t2_tse",
                "DP CS_SAG",
                "unknown",
                "unknown",
                "unknown",
            ],
            RepetitionTime=[None, None, None, "500", "3000", "3000"],
            EchoTime=[None, None, None, "10", "90", "30"],
        )
    )
    assert df["weight"].tolist() == ["T1", "T2", "PD", "T1", "T2", "PD"]
    assert df["fluid"].tolist() == [False, True, True, False, True, True]


def test_pixel_spacing_first_value() -> None:
    df = annotate(_meta(PixelSpacing=["0.3|0.31", None]))
    assert df["px"].iloc[0] == 0.3
    assert np.isnan(df["px"].iloc[1])


def test_laterality_prefers_tag_then_geometry() -> None:
    # 행 방향 +x, 열 방향 +y. 영상 중심 x = ipp_x + 0.5 * 100 px * 1 mm
    iop = "1|0|0|0|1|0"
    df = pd.DataFrame(
        {
            ID_COL: ["tagged", "right", "left", "centre"],
            "SeriesInstanceUID": ["a", "b", "c", "d"],
            "Laterality": ["L", None, None, None],
            "ImagePositionPatient": ["-200|0|0", "-200|0|0", "100|0|0", "-55|0|0"],
            "ImageOrientationPatient": [iop] * 4,
            "PixelSpacing": ["1|1"] * 4,
            "Rows": [100] * 4,
            "Columns": [100] * 4,
        }
    )
    assert laterality(df) == {"tagged": "L", "right": "R", "left": "L", "centre": None}
