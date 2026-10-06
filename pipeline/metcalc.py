"""Near-surface meteorology derivations for ERA5 (the Stage-1 bug-fix).

The original NCEP extraction took wind/pressure at upper-air levels and had no
V-wind, so wind direction was impossible. ERA5 provides 2 m T, 2 m dewpoint,
10 m U and V, and surface pressure, from which we derive the model's met features
correctly. All functions are pure (numpy/pandas) and self-tested below.

Conventions:
  - temperatures in degrees Celsius (ERA5 native is Kelvin -> subtract 273.15)
  - RH in %, wind speed in m/s, wind direction in meteorological degrees
    (0=N, 90=E, the direction the wind blows FROM), in [0, 360)
"""
from __future__ import annotations
import numpy as np
import pandas as pd

ZERO_C = 273.15


def rh_from_t_td(t_c, td_c):
    """Relative humidity (%) from temperature and dewpoint (deg C), August-Roche-Magnus."""
    a, b = 17.625, 243.04
    es = np.exp(a * t_c / (b + t_c))
    ed = np.exp(a * td_c / (b + td_c))
    return np.clip(100.0 * ed / es, 0, 100)


def spfh_from_td_p(td_c, p_pa):
    """Specific humidity (kg/kg) from dewpoint (deg C) and pressure (Pa)."""
    # saturation vapour pressure at dewpoint (Pa), Magnus over water
    e = 611.2 * np.exp(17.625 * td_c / (243.04 + td_c))
    return 0.622 * e / (p_pa - 0.378 * e)


def wind_speed(u, v):
    return np.hypot(u, v)


def wind_direction(u, v):
    """Meteorological wind direction (deg, FROM), 0=N, 90=E. Vector points TO (u,v)."""
    deg = (270.0 - np.degrees(np.arctan2(v, u))) % 360.0
    return deg


def to_beijing_daily(ds_hourly: pd.DataFrame, time_col: str, value_cols: list[str],
                     tz_offset_hours: int = 8, how: dict | None = None) -> pd.DataFrame:
    """Aggregate an hourly (UTC) long table to daily means in Beijing time.

    ds_hourly must have a UTC datetime column and the value columns. `how` maps a
    column to 'max' (e.g. gust); everything else uses 'mean'.
    Returns columns: date, <grouping cols preserved>, value_cols.
    """
    how = how or {}
    df = ds_hourly.copy()
    df[time_col] = pd.to_datetime(df[time_col]) + pd.Timedelta(hours=tz_offset_hours)
    df["date"] = df[time_col].dt.floor("D")
    keys = [c for c in df.columns if c not in value_cols + [time_col, "date"]]
    agg = {c: how.get(c, "mean") for c in value_cols}
    out = df.groupby(keys + ["date"], as_index=False).agg(agg)
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    return out


def derive_era5_features(t2m_k, d2m_k, u10, v10, sp_pa, gust=None) -> dict:
    """Map raw ERA5 fields -> the model's near-surface met features."""
    t_c = np.asarray(t2m_k) - ZERO_C
    td_c = np.asarray(d2m_k) - ZERO_C
    out = {
        "t2m": t_c,
        "rh": rh_from_t_td(t_c, td_c),
        "spfh": spfh_from_td_p(td_c, np.asarray(sp_pa)),
        "sp": np.asarray(sp_pa),
        "wind_speed": wind_speed(np.asarray(u10), np.asarray(v10)),
        "wind_dir": wind_direction(np.asarray(u10), np.asarray(v10)),
    }
    if gust is not None:
        out["gust"] = np.asarray(gust)
    return out


def _selftest():
    # RH: saturated air (T==Td) -> 100%
    assert abs(rh_from_t_td(20.0, 20.0) - 100.0) < 1e-6
    assert rh_from_t_td(20.0, 10.0) < 100
    # wind direction conventions
    assert abs(wind_direction(0, -1) - 0) < 1e-6      # wind from N (blows toward S, v=-1)
    assert abs(wind_direction(-1, 0) - 90) < 1e-6     # from E (blows toward W, u=-1)
    assert abs(wind_direction(0, 1) - 180) < 1e-6     # from S
    assert abs(wind_direction(1, 0) - 270) < 1e-6     # from W
    assert abs(wind_speed(3, 4) - 5) < 1e-9
    # spfh in plausible range
    q = spfh_from_td_p(10.0, 101325.0)
    assert 0 < q < 0.02
    # UTC+8 daily rollover: 23:00 UTC -> next Beijing day
    df = pd.DataFrame({"time": pd.to_datetime(["2020-01-01 23:00", "2020-01-01 10:00"]),
                       "gid": [1, 1], "t2m": [5.0, 7.0], "gust": [9.0, 3.0]})
    d = to_beijing_daily(df, "time", ["t2m", "gust"], 8, how={"gust": "max"})
    assert set(d["date"]) == {"2020-01-02", "2020-01-01"}
    print("metcalc self-test OK")


if __name__ == "__main__":
    _selftest()
