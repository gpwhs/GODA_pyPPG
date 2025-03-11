import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import os
import copy
from dotmap import DotMap
from scipy.interpolate import interp1d
import pyPPG
from pyPPG import PPG, Fiducials, Biomarkers
import pyPPG.preproc as PP
import pyPPG.fiducials as FP
import pyPPG.biomarkers as BM


class ParquetPulseWaveAnalyzer:
    """
    A class for analyzing pulse waves from PPG signals stored in a parquet file.

    This class adapts the functionality from pyPPG.validation.pw_anal.PulseWaveAnal
    to work with parquet files containing multiple PPG signals.
    """

    def __init__(self):
        """Initialize the analyzer."""
        self.OutData = {}
        self.M_FID_1 = pd.DataFrame()
        self.M_FID_2 = pd.DataFrame()

    def pw_extraction(
        self,
        ppg_signal: np.ndarray,
        subject_id: str = "unknown",
        fs: int = 125,
        filtering: bool = True,
        fL: float = 0,
        fH: float = 12,
        order: int = 4,
        sm_wins: Dict[str, int] = {"ppg": 50, "vpg": 10, "apg": 10, "jpg": 10},
        correction: pd.DataFrame = pd.DataFrame(),
        savefig: bool = False,
        savingfolder: str = "",
        show_fig: bool = False,
        print_flag: bool = False,
        heart_rate: Optional[float] = None,
        resample: bool = True,
    ) -> Tuple[DotMap, pd.DataFrame, Biomarkers]:
        """
        Extract features from a single PPG signal.

        Args:
            ppg_signal: Array containing the PPG signal
            subject_id: Identifier for the subject
            fs: Sampling frequency (Hz)
            filtering: Whether to apply filtering
            fL: Lower cutoff frequency (Hz)
            fH: Upper cutoff frequency (Hz)
            order: Filter order
            sm_wins: Dictionary of smoothing windows in milliseconds
            correction: DataFrame specifying fiducial point corrections
            savefig: Whether to save figures
            savingfolder: Folder to save figures to
            show_fig: Whether to show figures
            print_flag: Whether to print processing information
            heart_rate: Heart rate in beats per minute (used for resampling)
            resample: Whether to resample the signal from percentage to time-based

        Returns:
            Tuple of (signal, fiducial_points, biomarkers)
        """
        # Initialize the signal
        signal = DotMap()

        # Resample if needed (when signal is in percentage of cardiac cycle)
        if resample and heart_rate is not None:
            signal.v = self._resample_ppg(ppg_signal, heart_rate, source_fs, fs)
            if print_flag:
                print(
                    f"Resampled signal from {len(ppg_signal)} samples (% of cardiac cycle) to {len(signal.v)} samples at {fs}Hz"
                )
        else:
            signal.v = ppg_signal

        signal.fs = fs
        signal.name = subject_id

        # Preprocessing
        # Initialize the filters
        prep = PP.Preprocess(fL=fL, fH=fH, order=order, sm_wins=sm_wins)

        # Filter and calculate the PPG, PPG', PPG", and PPG'" signals
        signal.filtering = filtering
        signal.ppg, signal.vpg, signal.apg, signal.jpg = prep.get_signals(s=signal)

        # Create a PPG class
        signal.correction = correction
        s = PPG(signal, check_ppg_len=False)

        # Create a fiducial class
        fpex = FP.FpCollection(s)

        # Extract fiducial points
        ppg_fp = pd.DataFrame()
        ppg_fp["on"] = [0]
        ppg_fp["off"] = [len(signal.ppg) - 1]
        ppg_fp["sp"] = [np.argmax(signal.ppg)]

        peak = [ppg_fp.sp.iloc[0]]
        onsets = [ppg_fp.on.iloc[0], ppg_fp.off.iloc[0]]

        ppg_fp["dn"] = np.array(fpex.get_dicrotic_notch(peak, onsets))

        det_dn = np.array(fpex.get_dicrotic_notch(peak, onsets))
        vpg_fp = fpex.get_vpg_fiducials(onsets)
        apg_fp = fpex.get_apg_fiducials(onsets, peak)
        jpg_fp = fpex.get_jpg_fiducials(onsets, apg_fp)

        ppg_fp["dp"] = fpex.get_diastolic_peak(onsets, ppg_fp.dn, apg_fp.e)

        # Merge fiducials
        det_fp = self.merge_fiducials(ppg_fp, vpg_fp, apg_fp, jpg_fp)

        # Correct fiducials if correction DataFrame is provided
        if not correction.empty:
            det_fp = fpex.correct_fiducials(fiducials=det_fp, correction=correction)

        # Fill NaNs with zeros and convert to integers
        det_fp = det_fp.fillna(0).astype(int)

        # Create a copy with added onset for pulse boundary
        det_fp_new = pd.concat([det_fp, pd.DataFrame({"on": [ppg_fp.off.iloc[0]]})])
        det_fp_new = det_fp_new.fillna(0).astype(int)
        det_fp_new.index = [0, 1]

        if print_flag:
            print(det_fp)

        # Extract biomarkers
        bm = self.get_pw_bm(s=s, fp=det_fp_new)

        return s, det_fp, bm

    def merge_fiducials(self, ppg_fp, vpg_fp, apg_fp, jpg_fp):
        """
        Merge fiducial points from different PPG derivatives.

        Args:
            ppg_fp: Fiducial points from PPG
            vpg_fp: Fiducial points from PPG'
            apg_fp: Fiducial points from PPG"
            jpg_fp: Fiducial points from PPG'"

        Returns:
            DataFrame containing all fiducial points
        """
        fiducials = pd.DataFrame()
        for temp_sig in (ppg_fp, vpg_fp, apg_fp, jpg_fp):
            for key in list(temp_sig.keys()):
                fiducials[key] = temp_sig[key].values

        return fiducials

    def get_pw_bm(self, s: DotMap, fp: pd.DataFrame) -> Biomarkers:
        """
        Get pulse wave biomarkers.

        Args:
            s: Signal object
            fp: Fiducial points

        Returns:
            Biomarkers object
        """
        # Initialize the fiducials object
        fp_obj = Fiducials(fp=fp)

        # Initialize the biomarkers package
        bmex = BM.BmCollection(s=s, fp=fp_obj)

        # Extract biomarkers
        bm_defs, bm_vals = bmex.get_biomarkers(get_stat=False)

        # Create a biomarkers class
        bm = Biomarkers(bm_defs=bm_defs, bm_vals=bm_vals)

        return bm

    def extract_pw_feat(
        self,
        df: pd.DataFrame,
        ppg_column: str = "ppg",
        id_column: Optional[str] = None,
        hr_column: Optional[str] = None,
        fs: int = 125,
        savingfolder: str = "results",
        correction: pd.DataFrame = pd.DataFrame(),
        savefig: bool = False,
        show_fig: bool = False,
        print_flag: bool = True,
        resample: bool = False,
        source_fs: int = 100,
    ) -> Tuple[pd.DataFrame, Biomarkers]:
        """
        Extract pulse wave features from a DataFrame containing PPG signals.

        Args:
            df: DataFrame containing PPG data
            ppg_column: Name of the column containing PPG signals
            id_column: Name of the column containing subject identifiers
            hr_column: Name of the column containing heart rate values (used for resampling)
            fs: Sampling frequency (Hz)
            savingfolder: Folder to save results to
            correction: DataFrame specifying fiducial point corrections
            savefig: Whether to save figures
            show_fig: Whether to show figures
            print_flag: Whether to print processing information
            resample: Whether to resample signals from percentage to time-based
            source_fs: Number of samples in the source signal when resampling

        Returns:
            Tuple of (fiducial_points, biomarkers)
        """
        # Initialize default correction if not provided
        if correction.empty:
            correction = pd.DataFrame()
            corr_on = ["on", "v", "w", "f"]
            corr_off = ["dn", "dp"]
            correction.loc[0, corr_on] = True
            correction.loc[0, corr_off] = False

        # Create output folder
        os.makedirs(savingfolder, exist_ok=True)

        # Initialize result containers
        all_fp = pd.DataFrame()
        all_bm_vals = {}

        # Process each PPG signal
        number_of_rec = len(df)
        for i, (idx, row) in enumerate(df.iterrows()):
            # Get PPG signal
            ppg_signal = row[ppg_column]

            # Get subject ID
            subject_id = row[id_column] if id_column else f"subject_{idx}"

            # Get heart rate if available
            heart_rate = row[hr_column] if hr_column and hr_column in row else None

            # Extract features from the PPG signal
            s, fp, bm = self.pw_extraction(
                ppg_signal=ppg_signal,
                subject_id=subject_id,
                fs=fs,
                filtering=True,
                fL=0,
                fH=12,
                order=4,
                sm_wins={"ppg": 50, "vpg": 10, "apg": 10, "jpg": 10},
                correction=correction,
                savefig=savefig,
                savingfolder=savingfolder,
                show_fig=show_fig,
                print_flag=print_flag,
                heart_rate=heart_rate,
                resample=resample,
                source_fs=source_fs,
            )

            # Append fiducial points
            all_fp = pd.concat([all_fp, fp])

            # Append biomarker values
            for bm_key in bm.bm_vals.keys():
                if i == 0:
                    all_bm_vals[bm_key] = bm.bm_vals[bm_key]
                else:
                    all_bm_vals[bm_key] = pd.concat(
                        [all_bm_vals[bm_key], bm.bm_vals[bm_key]]
                    )

        # Update indices
        for bm_key in bm.bm_vals.keys():
            all_bm_vals[bm_key].index = range(0, number_of_rec)

        all_fp.index = range(0, number_of_rec)

        # Calculate statistics for all subjects
        from pyPPG.ppg_bm.statistics import get_statistics

        bm_stats = get_statistics(all_fp.sp, all_fp.on, all_bm_vals)

        # Create objects for return
        BMs = Biomarkers(bm_defs=bm.bm_defs, bm_vals=all_bm_vals, bm_stats=bm_stats)
        FPs = Fiducials(fp=all_fp)

        # Save data
        from pyPPG.datahandling import save_data

        save_data(
            s=s,
            fp=FPs,
            bm=BMs,
            savingformat="csv",
            savingfolder=savingfolder,
            print_flag=print_flag,
        )

        return FPs, BMs

    def process_parquet_file(
        self,
        parquet_path: str,
        ppg_column: str = "ppg",
        id_column: Optional[str] = None,
        hr_column: Optional[str] = None,
        fs: int = 125,
        output_dir: str = "results",
        resample: bool = False,
        source_fs: int = 100,
    ) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        """
        Process a parquet file containing PPG signals.

        Args:
            parquet_path: Path to the parquet file
            ppg_column: Name of the column containing PPG signals
            id_column: Name of the column containing subject identifiers
            hr_column: Name of the column containing heart rate values (for resampling)
            fs: Sampling frequency (Hz)
            output_dir: Directory to save the results
            resample: Whether to resample signals from percentage to time-based
            source_fs: Number of samples in the source signal when resampling

        Returns:
            Tuple of (fiducial_points_df, biomarker_values_dict, biomarker_stats_dict)
        """
        # Load the parquet file
        df = pd.read_parquet(parquet_path)

        # Initialize correction
        correction = pd.DataFrame()
        corr_on = ["on", "v", "w", "f"]
        corr_off = ["dn", "dp"]
        correction.loc[0, corr_on] = True
        correction.loc[0, corr_off] = False

        # Check if resampling is requested but HR column is missing
        if resample and (hr_column is None or hr_column not in df.columns):
            print(
                f"Warning: Resampling requested but HR column '{hr_column}' not found in data. Resampling will be skipped."
            )
            resample = False

        # Extract features
        fps, bms = self.extract_pw_feat(
            df=df,
            ppg_column=ppg_column,
            id_column=id_column,
            hr_column=hr_column,
            fs=fs,
            savingfolder=output_dir,
            correction=correction,
            savefig=True,
            show_fig=False,
            print_flag=True,
            resample=resample,
            source_fs=source_fs,
        )

        # Return results
        return fps.get_fp(), bms.bm_vals, bms.bm_stats


# Example usage:

if __name__ == "__main__":
    # Check if a parquet file is provided as an argument

    parquet_path = f"{os.getenv('BIOBANK_DATA_PATH')}/250k_waves_and_conditions.parquet"

    # Load the data to print columns
    df = pd.read_parquet(parquet_path)

    # using HR column, resample df["ppg"][1] to 250 Hz

    def resample_ppg(ppg_signal, heart_rate, target_fs=250):
        """
        Resamples a PPG signal to target_fs (default 250Hz).

        Parameters:
        - ppg_signal: np.array, PPG signal array of any length
        - heart_rate: float, in beats per minute (BPM)
        - target_fs: int, target sampling frequency (Hz)

        Returns:
        - resampled_signal: np.array, resampled PPG waveform
        """
        # Ensure HR is a valid float
        heart_rate = float(heart_rate)
        if heart_rate <= 0:
            raise ValueError("Heart rate must be a positive number.")

        # Compute heartbeat duration in seconds
        T = 60 / heart_rate

        # Get actual signal length
        signal_length = len(ppg_signal)

        # Define original and target time axes
        t_original = np.linspace(0, T, num=signal_length, endpoint=False)
        num_samples_target = int(
            T * target_fs
        )  # Total number of samples for resampled waveform
        t_resampled = np.linspace(
            0, T * (1 - 1e-6), num=num_samples_target, endpoint=False
        )  # Avoid exceeding bounds

        # Interpolation
        interpolator = interp1d(
            t_original,
            ppg_signal,
            kind="cubic",
            bounds_error=False,
            fill_value="extrapolate",
        )
        resampled_signal = interpolator(t_resampled)

        return resampled_signal

    signal = df["ppg"][1]
    resampled_signal = resample_ppg(signal, df["HR"][1], target_fs=250)

    np.savetxt(
        "Single_PW.csv", resampled_signal, delimiter=",", fmt="%.5f"
    )  # fmt controls precision
    # analyzer = ParquetPulseWaveAnalyzer()
    # # Process the file
    # fiducials, bm_values, bm_stats = analyzer.process_parquet_file(
    #     parquet_path=parquet_path,
    #     ppg_column="ppg",
    #     id_column="eid_visit",
    #     fs=250,
    #     output_dir="ppg_results",
    # )
    #
    # # Print summary
    # print(f"Processed {len(fiducials)} subjects")
    # print(f"Extracted {len(bm_values)} biomarker categories")
    #
    # # Display biomarker categories
    # print("Biomarker categories:", list(bm_values.keys()))
    #
    # # Save a combined biomarkers CSV for easier analysis
    # all_biomarkers = pd.DataFrame()
    #
    # # Start with an empty DataFrame with subject IDs
    # subject_ids = (
    #     fiducials["eid_visit"].unique()
    #     if "eid_visit" in fiducials.columns
    #     else range(len(fiducials))
    # )
    # all_biomarkers["eid_visit"] = subject_ids
    #
    # # Add the mean value of each biomarker for each subject
    # for category in bm_values:
    #     for column in bm_values[category].columns:
    #         if column in ["subject_id", "TimeStamp"]:
    #             continue
    #
    #         # Group by subject and calculate mean
    #         if "subject_id" in bm_values[category].columns:
    #             means = bm_values[category].groupby("subject_id")[column].mean()
    #             all_biomarkers[f"{category}_{column}"] = all_biomarkers[
    #                 "subject_id"
    #             ].map(means)
    #         else:
    #             # If no subject_id column, assume each row is a different subject
    #             all_biomarkers[f"{category}_{column}"] = bm_values[category][
    #                 column
    #             ].values
    #
    # # Save the combined biomarkers
    # all_biomarkers.to_parquet("ppg_results/all_biomarkers.parquet")
    # print(f"Saved combined biomarkers to ppg_results/all_biomarkers.parquet")
