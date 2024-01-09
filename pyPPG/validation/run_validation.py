import pyPPG.validation.pw_anal as PW
import pandas as pd
from datetime import datetime
import subprocess
import os

def run_matlab_script(script_folder, script, input_param):
    command = ['matlab', '-batch', f"eval('cd(\'\'{script_folder}\'\'); {script}(''{input_param}'')')"]
    subprocess.run(command, shell=False)

if __name__ == '__main__':

    # Initialise the pulse wave package
    pwex = PW.PulseWaveAnal()

    # Initialise the correction
    correction = pd.DataFrame()
    corr_on = ['on', 'v', 'w', 'f']
    corr_off = ['dn', 'dp']
    correction.loc[0, corr_on] = True
    correction.loc[0, corr_off] = False

    # Save time for the results of the output folders
    date = datetime.now()
    dname = str(date.year) + '_' + str(date.month) + '_' + str(date.day) + '_' + str(date.hour) + '_' + str(date.minute)

    # Run PPG-BP Evaluation for annotation and pyPPG
    pwex.eval_PPG_BP(plts=True, correction=correction, dname=dname)

    # Command to run MATLAB script for PPGFeat
    current_directory = os.getcwd()
    script_folder = current_directory+os.sep+'PPGFeat'
    scipt='get_PPGFeat_fps'
    run_matlab_script(script_folder,scipt,dname)

    # Command to run MATLAB script for PPGFeat
    current_directory = os.getcwd()
    script_folder = current_directory+os.sep+'PulseAnalyse'
    scipt='get_PA_fps'
    run_matlab_script(script_folder,scipt,dname)

    # Run Benchmarking
    pwex.benchmark_PPG_BP(detector='pyPPG', dname=dname, plt=True, prnt=False)
    pwex.benchmark_PPG_BP(detector='PPGFeat', dname=dname, plt=True, prnt=False)
    pwex.benchmark_PPG_BP(detector='PulseAnal', dname=dname, plt=True, prnt=False)

    print('End of PPG Analysis and Benchmarking!')