from typing import Dict, List, Union, Optional, Tuple, Any
import os
import numpy as np
import pandas as pd
from dotmap import DotMap
import re
from scipy import interpolate
import warnings

import pyPPG
from pyPPG import PPG, Fiducials, Biomarkers
import pyPPG.preproc as PP
import pyPPG.fiducials as FP
import pyPPG.biomarkers as BM
import pyPPG.ppg_sqi as SQI


def extract_single_pulse_biomarkers(
    ppg_signal: np.ndarray,
    fs: int,
    filtering: bool = True,
    fL: float = 0.5000001,
    fH: float = 12,
    order: int = 4,
    sm_wins: Dict[str, int] = {"ppg": 50, "vpg": 10, "apg": 10, "jpg": 10},
) -> Tuple[Dict[str, Any], float]:
    """
    Extract biomarkers from a single pulse wave PPG signal.
    Based on the pw_extraction method in pyPPG.validation.pw_anal.

    Args:
        ppg_signal: Single pulse wave PPG signal
        fs: Sampling frequency
        filtering: Whether to apply filtering
        fL: Lower cutoff frequency
        fH: Upper cutoff frequency
        order: Filter order
        sm_wins: Smoothing windows for each signal type

    Returns:
        (biomarkers_dict, sqi): Tuple of biomarkers dictionary and SQI value
    """
    # Create signal object
    signal = DotMap()
    signal.v = ppg_signal
    signal.fs = fs
    signal.name = "single_pulse"

    # Initialize preprocessing
    prep = PP.Preprocess(fL=fL, fH=fH, order=order, sm_wins=sm_wins)

    # Filter and calculate signals
    signal.filtering = filtering
    signal.fL = fL
    signal.fH = fH
    signal.order = order
    signal.sm_wins = sm_wins
    signal.ppg, signal.vpg, signal.apg, signal.jpg = prep.get_signals(s=signal)

    # Set up correction
    correction = pd.DataFrame()
    corr_on = ["on", "v", "w", "f"]
    correction.loc[0, corr_on] = True
    signal.correction = correction

    # Create PPG class
    s = PPG(s=signal, check_ppg_len=False)

    # Create fiducials instance
    fpex = FP.FpCollection(s=s)

    # Extract fiducial points
    ppg_fp = pd.DataFrame()
    ppg_fp["on"] = [0]
    ppg_fp["off"] = [len(signal.ppg) - 1]
    ppg_fp["sp"] = [np.argmax(signal.ppg)]

    peak = [ppg_fp.sp.iloc[0]]
    onsets = [ppg_fp.on.iloc[0], ppg_fp.off.iloc[0]]

    # Get dicrotic notch
    ppg_fp["dn"] = np.array(fpex.get_dicrotic_notch(peak, onsets))

    # Get derivative-based fiducial points
    vpg_fp = fpex.get_vpg_fiducials(onsets)
    apg_fp = fpex.get_apg_fiducials(onsets, peak)
    jpg_fp = fpex.get_jpg_fiducials(onsets, apg_fp)

    # Get diastolic peak
    ppg_fp["dp"] = fpex.get_diastolic_peak(onsets, ppg_fp.dn, apg_fp.e)

    # Merge all fiducial points
    fiducials = pd.DataFrame()
    for temp_sig in (ppg_fp, vpg_fp, apg_fp, jpg_fp):
        for key in list(temp_sig.keys()):
            temp_val = temp_sig[key].values
            fiducials[key] = [None] if len(temp_val) == 0 else temp_val

    # Correct fiducial points
    fiducials = fpex.correct_fiducials(fiducials, correction)

    # Create fiducials class
    fiducials = fiducials.fillna(0).astype("Int64")
    fiducials = pd.concat([fiducials, pd.DataFrame({"on": [ppg_fp.off.iloc[0]]})])
    fiducials.index = range(len(fiducials))
    fp = Fiducials(fp=fiducials)

    # Calculate SQI
    try:
        sqi = np.mean(SQI.get_ppgSQI(ppg=s.ppg, fs=s.fs, annotation=fp.sp)) * 100
    except:
        sqi = np.nan

    # Extract biomarkers
    bmex = BM.BmCollection(s=s, fp=fp)
    bm_defs, bm_vals = bmex.get_biomarkers(get_stat=False)

    # Create a flattened dictionary of biomarkers
    biomarkers_dict = {}
    for category in bm_vals:
        for col in bm_vals[category].columns:
            if col != "TimeStamp":
                # Clean column name
                clean_name = re.sub(r"[\/\(\)\-\%]", "_", f"{category}_{col}")
                clean_name = re.sub(r"_+", "_", clean_name).strip("_")

                # Get biomarker value
                if len(bm_vals[category]) > 0:
                    biomarkers_dict[clean_name] = bm_vals[category][col].iloc[0]
                else:
                    biomarkers_dict[clean_name] = np.nan

    return biomarkers_dict, sqi


def extract_ppg_features(
    df: pd.DataFrame,
    ppg_column: str = "ppg",
    hr_column: str = "HR",
    target_fs: int = 250,
    filtering: bool = True,
    fL: float = 0.5000001,
    fH: float = 12,
    order: int = 4,
    sm_wins: Dict[str, int] = {"ppg": 50, "vpg": 10, "apg": 10, "jpg": 10},
    store_resampled: bool = False,
) -> pd.DataFrame:
    """
    Extract biomarkers and SQI from pulse wave PPG signals in a dataframe.

    Args:
        df: DataFrame containing PPG signals
        ppg_column: Column name containing the PPG signal arrays
        hr_column: Column name containing heart rate values (BPM)
        target_fs: Target sampling frequency for resampling
        filtering: Whether to apply filtering to the signal
        fL: Lower cutoff frequency (Hz)
        fH: Upper cutoff frequency (Hz)
        order: Filter order
        sm_wins: Dictionary of smoothing windows in milliseconds
        store_resampled: Whether to store resampled signals in the result

    Returns:
        DataFrame with added biomarker columns and SQI
    """
    # Create a copy of the dataframe
    result_df = df.copy()

    # Initialize lists for results
    biomarkers_data = []
    resampled_signals = []
    sqi_values = []

    # Process each PPG signal
    for i, row in df.iterrows():
        try:
            # Get the PPG signal
            ppg_signal = row[ppg_column]

            # Skip if signal is empty or all NaN
            if len(ppg_signal) == 0 or np.isnan(ppg_signal).all():
                print(f"Skipping subject {i}: empty or all NaN signal")
                biomarkers_data.append({})
                sqi_values.append(np.nan)
                resampled_signals.append(np.array([]))
                continue

            # Get heart rate (use a default if not provided)
            hr = (
                row[hr_column]
                if hr_column in df.columns and not pd.isna(row[hr_column])
                else 60
            )

            # Resample the signal based on heart rate
            cycle_duration = 60.0 / hr  # seconds
            x_percent = (
                np.linspace(0, 100, len(ppg_signal))
                if np.max(ppg_signal) > 5
                else np.linspace(0, 1, len(ppg_signal))
            )
            x_time = x_percent * cycle_duration / np.max(x_percent)
            num_samples = int(cycle_duration * target_fs)
            new_time_points = np.linspace(0, cycle_duration, num_samples)

            # Interpolate signal
            interpolator = interpolate.interp1d(
                x_time,
                ppg_signal,
                kind="cubic",
                bounds_error=False,
                fill_value="extrapolate",
            )
            resampled_ppg = interpolator(new_time_points)
            resampled_signals.append(resampled_ppg)

            # Extract biomarkers and SQI
            try:
                bm_dict, sqi = extract_single_pulse_biomarkers(
                    ppg_signal=resampled_ppg,
                    fs=target_fs,
                    filtering=filtering,
                    fL=fL,
                    fH=fH,
                    order=order,
                    sm_wins=sm_wins,
                )
                biomarkers_data.append(bm_dict)
                sqi_values.append(sqi)
            except Exception as e:
                print(f"Biomarker extraction error for subject {i}: {str(e)}")
                biomarkers_data.append({})
                sqi_values.append(np.nan)

        except Exception as e:
            print(f"Error processing signal for subject {i}: {str(e)}")
            biomarkers_data.append({})
            sqi_values.append(np.nan)
            resampled_signals.append(np.array([]))

    # Add SQI to result dataframe
    result_df["ppg_sqi"] = sqi_values

    # Combine all biomarker dictionaries into dataframe columns
    all_keys = set()
    for bm_dict in biomarkers_data:
        all_keys.update(bm_dict.keys())

    for key in all_keys:
        result_df[key] = [bm_dict.get(key, np.nan) for bm_dict in biomarkers_data]

    # Optionally store resampled signals
    if store_resampled:
        result_df["resampled_ppg"] = resampled_signals

    return result_df


def main():
    wave_data = pd.read_parquet(
        f"{os.getenv('BIOBANK_DATA_PATH')}/250k_waves_and_conditions.parquet"
    )
    wave_data_with_pyppg_features = extract_ppg_features(wave_data)

    for col in wave_data_with_pyppg_features.columns:
        print(col)


if __name__ == "__main__":
    main()
