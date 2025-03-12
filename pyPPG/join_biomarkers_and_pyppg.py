import pandas as pd
import os
import pickle



def main():

    data = pd.read_parquet(
        f"{os.getenv('BIOBANK_DATA_PATH')}/250k_waves_and_conditions.parquet"
    )
    data = data.dropna()
    data.reset_index(drop=True, inplace=True)
    with open("validation/bm_fp.pkl", "rb") as f:
                bms, fps = pickle.load(f)
    print("Joining dataframes")
    bm_vals = bms.bm_vals
    ppg_sig_bms = bm_vals['ppg_sig'].drop(columns=['TimeStamp'])
    sig_ratio_bms = bm_vals['sig_ratios'].drop(columns=['TimeStamp'])
    ppg_deriv_bms = bm_vals['ppg_derivs'].drop(columns=['TimeStamp'])
    deriv_ratio_bms = bm_vals['derivs_ratios'].drop(columns=['TimeStamp'])
    full_data = pd.concat([data, ppg_sig_bms, sig_ratio_bms, ppg_deriv_bms, deriv_ratio_bms], axis=1)
    full_data = full_data.drop(columns=['Tpp'])
    print("Saving joined dataframes")
    data.to_parquet(f"{os.getenv('BIOBANK_DATA_PATH')}/250k_waves_conditions_pyppgfeatures.parquet")



if __name__ == "__main__":
    main()

