'''
This file is part of the Pareto_optimal diagnosis module of the MDTF code package (see mdtf/MDTF-diagnostics/LICENSE.txt).

DESCRIPTION:

REQUIRED MODULES:

AUTHORS: Xianan Jiang, David Neelin (UCLA)

LAST EDIT:

REFERENCES: 

'''

import os
import subprocess
import time
import numpy

# call ncl library via subprocess
def generate_ncl_plots(nclPlotFile):
    # check if the nclPlotFile exists - 
    # don't exit if it does not exists just print a warning.
    try:
        pipe = subprocess.Popen(['ncl {0}'.format(nclPlotFile)], shell=True, stdout=subprocess.PIPE)
        output = pipe.communicate()[0].decode()
        print('NCL routine {0} \n {1}'.format(nclPlotFile,output))
        while pipe.poll() is None:
            time.sleep(0.5)
    except OSError as e:
        print('WARNING',e.errno,e.strerror)
    return 0

missing_file=0
#============================================================
if missing_file==1:
    print("Pareto Optimal Diagnostic Package will NOT be executed!")
else:

    # ======================================================================
    try:
        os.system("python "+os.environ["POD_HOME"]+"/"+"pareto_calculation_parameters.py")
    except OSError as e:
        print('WARNING',e.errno,e.strerror)
        print("**************************************************")
        print("Parameter Setting (pareto_calculation_parameters.py) is NOT Executed as Expected!")
        print("**************************************************")

    # ======================================================================
    # Process model output from the Targetting GCM
    # 1) Calculate winter mean fields; 2) Interpolated to standard grids
    try:
        os.system("python "+os.environ["POD_HOME"]+"/"+"process_model_data.py")
    except OSError as e:
        print('WARNING',e.errno,e.strerror)
        print("**************************************************")
        print("Target Model Data Processing (process_model_data.py) is NOT Executed as Expected!")
        print("**************************************************")

    # ======================================================================
    # Read preprocessed OBS and CMIP6 model data, and target GCM data
    try:
        os.system("python "+os.environ["POD_HOME"]+"/"+"input_data.py")
    except OSError as e:
        print('WARNING',e.errno,e.strerror)
        print("**************************************************")
        print("Prepration of input data from OBS and CMIP6 GCMs (input_data.py) is NOT Executed as Expected!")
        print("**************************************************")

    ## ======================================================================
    ## Conduct Pareto Optimal analysis for the three variables over speficied region
    ##  K: update 3
    try:
        os.system("python "+os.environ["POD_HOME"]+"/"+"calculate_optimal_front.py")
    except OSError as e:
        print('WARNING',e.errno,e.strerror)
        print("**************************************************")
        print("Pareto Optimal Front Calculations (calculate_optimal_front.py) is NOT Executed as Expected!")
        print("**************************************************")

    # ======================================================================
    # Plot biases in the winter mean state related to the three variables over regions specified
    try:
        os.system("python "+os.environ["POD_HOME"]+"/"+"figure1_plot_winter_mean_climate.py")
    except OSError as e:
        print('WARNING',e.errno,e.strerror)
        print("**************************************************")
        print("Plotting Winter Mean patterns (Fg. 1) (figure1_plot_winter_mean_climate.py) is NOT Executed as Expected!")
        print("**************************************************")

    ## ======================================================================
    ## Plot results of Pareto Optimal analysis including both 2D and 3D
    try:
        os.system("python "+os.environ["POD_HOME"]+"/"+"figure2_plot_optimal_fronts.py")
    except OSError as e:
        print('WARNING',e.errno,e.strerror)
        print("**************************************************")
        print("Plotting Pareto Optimal Fronts (Fig. 2) (figure2_plot_optimal_fronts.py) is NOT Executed as Expected!")
        print("**************************************************")

    ## ======================================================================
    ## PLotting spatial winter mean patterns for model combos with minimum RMSEs in each of the three variables
    try:
        os.system("python "+os.environ["POD_HOME"]+"/"+"figure3_optimal_winter_mean_patterns.py")
    except OSError as e:
        print('WARNING',e.errno,e.strerror)
        print("**************************************************")
        print("Plotting Optimal Winter Mean Patterns (Fig. 3) (figure3_optimal_winter_mean_patterns.py) is NOT Executed as Expected!")
        print("**************************************************")

    para = numpy.load(os.environ["WK_DIR"]+"/pareto_parameters.npy",allow_pickle=True)
    exp_name=para[()]["exp_name"]
    if exp_name=='CA':
       os.system("cp "+os.environ["POD_HOME"]+"/Pareto_optimal.html.CA "+os.environ["POD_HOME"]+"/Pareto_optimal.html")
    if exp_name=='SAM':
       os.system("cp "+os.environ["POD_HOME"]+"/Pareto_optimal.html.SAM "+os.environ["POD_HOME"]+"/Pareto_optimal.html")

#   os.system("cp *.pdf /temp/synology")  # this is only for testing purpose
    os.system("rm "+os.environ["WK_DIR"]+"/*.txt "+os.environ["WK_DIR"]+"/*.npy "+os.environ["WK_DIR"]+"/*.pdf")
    os.system("cp "+os.environ["POD_HOME"]+"/MDTF_Documentation_pareto_optimal.pdf "+os.environ["WK_DIR"]+"/")

    print("**************************************************")
    print("Pareto Optimal Diagnostic Package Executed!")
    print("**************************************************")
