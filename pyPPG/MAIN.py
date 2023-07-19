from DataHandling import*
from Prefiltering import*
# from FiducialPoints import*
import FiducialPoints as Fp
from Biomarkers import*
from Statistics import*

import matplotlib.pyplot as plt
import numpy as np
from dotmap import DotMap
import time

###########################################################################
################################### MAIN ##################################
###########################################################################
if __name__ == '__main__':

    ## Load data
    s=load_data(filtering=True)

    ## Get Fiducials Points
    # fiducials = getFiducialPoints(s,correct=True)
    fp = Fp.FiducialPoints(s)
    fiducials=fp.getFiducialPoints(correct=True)

    ## Plot Fiducials Points
    plot_fiducials(s, fiducials,savefig=True)

    ## Get Fiducials Biomarkers, Summary and Statistics
    ppg_biomarkers = Biomarkers(s, fiducials)
    ppg_statistics = Statistics(fiducials['sp'], fiducials['on'], ppg_biomarkers)

    ## Save data
    save_data(s,fiducials,ppg_biomarkers,ppg_statistics)

    print('Program finished')
