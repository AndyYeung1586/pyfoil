import glob
import subprocess
import random
import time
import os
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from multiprocess import pool

import fit_spline as fs


# declare a dataclass to pass xfoil parameters easier
@dataclass
class XFoilConfig:
    airfoil_path: str
    mach: float
    re: float
    ncrit: int
    panels: int
    ifile: str
    ofile: str
    start_alfa: float
    end_alfa: float
    alfa_step: float
    panel_recon: bool
    tripped: bool
    verbose: int
    timeout: int


def main():
    # ================================ Parameters ================================
    airfoil_coords_path = 'Airfoil'
    n_subprocess = 5
    mach = 0.3
    re = 2e6
    ncrit = 1
    polar_folder = f'N{ncrit}'
    verbose = 1

    # ============================== Compile Parameters ==============================
    airfoil_paths = glob.glob(f'{airfoil_coords_path}/*.txt')
    airfoil_names = get_airfoil_names(airfoil_paths)
    airfoil_sublist = shuffle_airfoils(airfoil_names, n_subprocess)
    params = [
        (airfoils, mach, re, ncrit, polar_folder, 199, verbose)
        for airfoils in airfoil_sublist
    ]

    # ================================ Run XFoil ================================
    # run xfoil given the parameters (baseline_airfoils, mach, re, ncrit, pert, polar, panels)
    polar_output_init(polar_folder)
    with pool.Pool(processes=len(params)) as p:
        p.starmap(run_xfoil, params)

    # run targeted airfoil cases
    # run_xfoil(['SSCA09_0019', 'SSCA09_0029'], mach, re, ncrit, 'test', verbose=4)

    # ================================ Process data ================================
    # process xfoil data after running xfoil
    extract_polar(polar_folder, mach, re, ncrit)
    return


def get_airfoil_names(airfoil_paths):
    airfoil_names = []
    for airfoil_path in airfoil_paths:
        airfoil_name = airfoil_path.split('\\')[-1].split('.')[0]
        airfoil_names.append(airfoil_name)
    return airfoil_names


def shuffle_airfoils(airfoil_names, n_sublist):
    random.shuffle(airfoil_names)

    sublist_size = len(airfoil_names) // n_sublist
    if sublist_size == 0:
        raise ValueError('n_sublist is larger than the list length!')
    sublist_remainder = len(airfoil_names) % n_sublist

    start = 0
    airfoil_sublist = []
    for i in range(n_sublist):
        if sublist_remainder >= 1:
            end = start + sublist_size + 1
            airfoil_sublist.append(airfoil_names[start:end])
            sublist_remainder -= 1
            start = end
        else:
            end = start + sublist_size
            airfoil_sublist.append(airfoil_names[start:end])
            start = end
    return airfoil_sublist


# ====================================================================================
# ==================================  running xfoil  =================================
# ====================================================================================
def run_xfoil(airfoils, mach, re, ncrit, polar=None, panels=199, verbose=1, timeout=45):
    # initialize variables
    set_fail = 0
    airfoil_missed = []

    # initialize input file and output folder
    if polar is None:
        polar = f'M{mach}_Re{re}_N{ncrit}'
    polar_output_init(polar)

    for airfoil in airfoils:
        # create a config dataclass for each airfoil
        xfoil_cfg = XFoilConfig(
            airfoil_path=f'Airfoil/{airfoil}.txt',
            mach=mach,
            re=re,
            ncrit=ncrit,
            panels=panels,
            ifile=f'{airfoil}_{mach}_{re}_{ncrit}_{panels}.txt',
            ofile=f'Polar/{polar}/{airfoil}.txt',
            start_alfa=0,
            end_alfa=20,
            alfa_step=0.2,
            panel_recon=False,
            tripped=False,
            verbose=verbose,
            timeout=timeout
        )

        # ==================== First Pass ====================
        vprint(verbose, 2, f'Running {airfoil}')

        try:
            # create a xfoil instruction file
            input_init(xfoil_cfg.ifile)

            # # call xfoil to run full polar based on the config params, this run orginal coordinates (not rec)
            # if not polar_complete(xfoil_cfg.ofile, 2):
            #     run_sweep(xfoil_cfg, 0, 20, 0.2)
            #     run_sweep(xfoil_cfg, 0, -10, -0.2)

            # ==================== panel number sweep ====================
            def run_case():
                for n_panels in [160, 199, 299]:
                    for alfa_step in [0.2, 0.1]:
                        if polar_complete(xfoil_cfg.ofile, 2):
                            vprint(verbose, 1, f'Polar complete for {airfoil}!')
                            return 1

                        if os.path.exists(xfoil_cfg.ofile):
                            # remove old polar folder
                            vprint(verbose, 2, 'Removing old polar file ...')
                            os.remove(xfoil_cfg.ofile)

                        # rerun xfoil with new config params
                        vprint(verbose, 2, f'Running {airfoil} with {n_panels} panels ...')
                        new_panel_cfg = replace(xfoil_cfg, panels=n_panels, panel_recon=True)
                        run_sweep(new_panel_cfg, 0, 20, alfa_step)
                        run_sweep(new_panel_cfg, 0, -10, -alfa_step)

                return 0
            case_done = run_case()

            if not case_done:
                vprint(verbose, 1, f'\tTOTAL FAILURE FOR {airfoil}')
                airfoil_missed.append(airfoil)
                set_fail += 1

                if os.path.exists(xfoil_cfg.ofile):
                    # remove old polar folder
                    vprint(verbose, 2, 'Removing old polar file ...')
                    os.remove(xfoil_cfg.ofile)

        finally:
            try:
                # remove input file
                os.remove(xfoil_cfg.ifile)
            except Exception:
                print('YIKES')
                pass

    if verbose >= 1:
        print(f'all missed airfoil: \n{airfoil_missed}')
        print(f'sets failed: {set_fail}\n')
    return


def vprint(verbose, level, message):
    if verbose >= level:
        print(message)
    return


def polar_output_init(polar):
    """
    Check if output directory for polar files exist, and create directory if needed

    :param polar: name of the output directory
    :return:
    """
    output_dir = f'Polar/{polar}'
    directories = [x[0] for x in os.walk('Polar/')]

    if output_dir not in directories:
        print('Making Model Directory')
        os.mkdir(output_dir)
    return


def input_init(ifile):
    """
    initialize a parameterless xfoil script

    :param ifile: xfoil script name
    :return:
    """
    # initialize input file, line []
    f = open(f'{ifile}', 'w')
    f.write('plop\n')                               # toggle plotting option [0]
    f.write('g f\n\n')                              # turn off graphics [1]

    f.write(f'load (Airfoil/name.txt)\n\n')           # load airfoil coords [3]
    f.write('pane\n')                               # panel menu [5]
    f.write('ppar\n')                               # toggle paneling menu [6]
    f.write(f'n (panels)\n\n\n')                      # set panel number [7]

    f.write("oper\n")                               # operation menu [10]
    f.write(f'm (mach)\n')                            # set mach number [11]
    f.write(f'visc (re)\n')                           # set to viscous mode and Re [12]
    f.write('vpar\n')                               # toggle bl parameters menu [13]
    f.write(f'N 9\n\n')                             # set NCrit (1-9) [14]

    f.write('pacc\n')                               # toggle point accumulator to active polar [16]
    f.write(f'(Polar/M0.1/airfoil_name.txt)\n\n')     # save polar file [17]

    f.write(f'aseq 0 20 step\n\n')                  # set AoA sweep [19]
    f.write('quit\n')                               # quit script [21]
    f.close()
    return


def check_airfoil(airfoil_path):
    """
    check for airfoil with negative thickness, return true if such
    """
    df = pd.read_csv(airfoil_path, header=None, sep='\s+', names=['x/c', 'y/c'], skiprows=1)
    split = df['x/c'].idxmin()
    lwr_curve = df.iloc[:split + 1].iloc[::-1]
    lwr_curve.reset_index(drop=True, inplace=True)
    upr_curve = df.iloc[split:]
    upr_curve.reset_index(drop=True, inplace=True)

    thickness = upr_curve['y/c']-lwr_curve['y/c']
    return (thickness < 0).any().any()


def run_sweep(cfg: XFoilConfig, start_alfa, end_alfa, alfa_step):
    cfg.start_alfa, cfg.end_alfa, cfg.alfa_step = start_alfa, end_alfa, alfa_step

    run_done = False
    repeat, previous_check, tolerance = 0, 99, 2
    while not run_done:
        vprint(cfg.verbose, 3, 'Calling XFoil ... ')
        call_xfoil(cfg)
        last_alfa = find_last_alfa(cfg.ofile, cfg.end_alfa)
        vprint(cfg.verbose, 3, f'Last converged angle: {last_alfa}')

        # this check if the solver diverged at the same AoA
        if last_alfa == previous_check:
            vprint(cfg.verbose, 3, 'Repeated!')
            repeat += 1
        else:
            repeat = 0
        previous_check = last_alfa

        # this check if too much cases have failed
        if (repeat+1) * abs(cfg.alfa_step) > tolerance:
            vprint(cfg.verbose, 2, f'FAILED TO COMPLETE SWEEP FOR {cfg.airfoil_path}!')
            run_done = True
            continue

        # this determines if the run is complete or AoAs need to be adjusted
        if abs(last_alfa) >= abs(end_alfa):
            vprint(cfg.verbose, 3, f'RUN COMPLETED!')
            run_done = True
            continue
        else:
            cfg.start_alfa = round(last_alfa + (repeat + 1) * cfg.alfa_step, 1)
            cfg.end_alfa = round(end_alfa + (repeat + 1) * cfg.alfa_step, 1)
            vprint(cfg.verbose, 3, f'RERUNNING ... now restart angle: {cfg.start_alfa} to {cfg.end_alfa}')

    return


def call_xfoil(cfg: XFoilConfig):
    """
    given simulation parameters, modify the xfoil script and call xfoil.exe (windows only)
    """
    # making input file for positive AoA
    with open(f'{cfg.ifile}', 'r+') as file:
        lines = file.readlines()
        lines[3] = f'load {cfg.airfoil_path}\n'
        if cfg.panel_recon:
            lines[5] = 'pane\n'
            lines[6] = 'ppar\n'
            lines[7] = f'n {cfg.panels}\n'
        else:
            lines[5] = '\n'
            lines[6] = '\n'
            lines[7] = '\n'
        lines[11] = f'm {cfg.mach}\n'

        re = cfg.re * 0.000001  # for some bizarre reason, Re number in polar dump file ALWAYS output as X.XXXe6
        lines[12] = f'visc {re:.3f}e+06\n'
        lines[14] = f'xtr 0 0\n' if cfg.tripped else f'N {cfg.ncrit}\n'
        lines[17] = f'{cfg.ofile}\n'
        lines[19] = f'aseq {round(cfg.start_alfa, 1)} {round(cfg.end_alfa, 1)} {cfg.alfa_step}\n'

        file.seek(0)
        file.writelines(lines)
        file.truncate()

    # call xfoil (mute xfoil terminal output)
    try:
        subprocess.run(f"xfoil.exe < {cfg.ifile} > NUL", shell=True, timeout=cfg.timeout)
        # subprocess.call(f"xfoil.exe < {cfg.ifile}", shell=True)
    except subprocess.TimeoutExpired:
        print(f'**** {cfg.airfoil_path} Timed Out! ****')

    return


def find_last_alfa(ofile, intended_alfa):
    """
    given polar output file, find the last converged angle of attck
    :param ofile: polar output name
    :param intended_alfa: last intended angle of attack
    :return: the last converged angle of attack
    """
    # Read the file and process the lines
    last_alfa, min_residual = 0, 99
    with open(ofile, 'r') as file:
        lines = file.readlines()[12:]  # Skip to the polar table
        for line in lines:
            # Split the line into columns and get the alfa value (first column)
            try:
                alfa = float(line.split()[0])
                residual = abs(intended_alfa - alfa)
                if residual < min_residual:
                    last_alfa = alfa
            except (IndexError, ValueError):
                continue
    return last_alfa


def find_missing_alfa(ofile):
    """
    given polar output file, report any missing angle between [-10, 20]
    :param ofile: polar output name
    :return: a set of missing angles
    """
    # initialize expected and found sets
    expected_alfas = set(range(-10, 21))
    found_alfas = set()

    try:
        # Read the file and process the lines
        with open(ofile, 'r') as file:
            lines = file.readlines()[12:]  # Skip to the polar table
            for line in lines:
                # Split the line into columns and get the alfa value (first column)
                try:
                    alfa = float(line.split()[0])
                    if int(alfa) == alfa:  # Check if alfa is an integer value
                        found_alfas.add(int(alfa))
                except (IndexError, ValueError):
                    continue
    except FileNotFoundError:
        return expected_alfas

    # return the missing alfa values
    return expected_alfas - found_alfas


def polar_complete(ofile, tolerance):
    missing_alfas = find_missing_alfa(ofile)
    if len(missing_alfas) == 0:
        return True

    try:
        file = open(ofile, 'r')
        lines = file.readlines()[12:]  # Skip to the polar table
    except FileNotFoundError:
        return False

    for missing_alfa in missing_alfas:
        upper_check = 99.
        lower_check = -99.
        for line in lines:
            dalfa = float(line.split()[0]) - missing_alfa
            if 0 < dalfa < upper_check:
                upper_check = dalfa
            elif 0 > dalfa > lower_check:
                lower_check = dalfa

        # check if the margin is small enough that linear interpolation make sense
        if upper_check - lower_check > tolerance:
            file.close()
            return False

    file.close()
    return True


def interpolate_data(missing_alfa, upper_data, lower_data):
    case = [missing_alfa]
    for i in range(1, len(upper_data)):
        coeff = (upper_data[i]-lower_data[i])/(upper_data[0]-lower_data[0])*(missing_alfa-lower_data[0])+lower_data[i]
        case.append(coeff)
    return case


def camber_thickness(chebyshev_poly, deg):
    # this turn upr_c into camber polynomial and lwr_c into thickness polynomial
    for i in range(deg+1):
        upr_c, lwr_c = chebyshev_poly[f'upr_c{i}'], chebyshev_poly[f'lwr_c{i}']
        chebyshev_poly[f'upr_c{i}'] = (upr_c + lwr_c)/2
        chebyshev_poly[f'lwr_c{i}'] = upr_c - lwr_c
    return chebyshev_poly


# ====================================================================================
# ======================  extracting xfoil polar (vectorized)  =======================
# ====================================================================================
def extract_polar(polar_folder, mach, re, ncrit):
    # input file parameters
    alfa_range = np.linspace(-10, 20, 31)
    deg = 8
    recovered = 0
    upper_data = []
    lower_data = []
    df = pd.DataFrame()
    airfoils = get_airfoil_names(glob.glob(f'Polar/{polar_folder}/*.txt'))

    # extract all info for each airfoil
    for airfoil in airfoils:
        # initialize
        check_set = set()
        cases = []

        # Read the file and process the lines
        ofile = f'Polar/{polar_folder}/{airfoil}.txt'
        file = open(ofile, 'r')
        lines = file.readlines()[12:]  # Skip to the polar table

        # first for loop to gather all integer alfa cases
        for line in lines:
            data = get_data_point(line)
            alfa = data[0]
            if alfa == int(alfa) and alfa not in check_set and alfa in alfa_range:
                check_set.add(alfa)
                cases.append(data)

        # Second for loop to attempt to interpolate data for the missing alfa cases
        missing_alfas = find_missing_alfa(ofile)
        for missing_alfa in missing_alfas:
            print(f'Now checking {airfoil}, @ Aoa:{missing_alfa}')
            upper_check = 99.
            lower_check = -99.
            for line in lines:
                dalfa = float(line.split()[0]) - missing_alfa
                if 0 < dalfa < upper_check:
                    upper_check = dalfa
                    upper_data = get_data_point(line)
                elif 0 > dalfa > lower_check:
                    lower_check = dalfa
                    lower_data = get_data_point(line)

            # check if the margin is small enough that linear interpolation make sense
            if upper_check - lower_check < 2.0:
                print('linear interpolate')
                cases.append(interpolate_data(missing_alfa, upper_data, lower_data))
                recovered += 1
            else:
                print('Margin too large!')
                # since we want a vectorized dataset, incomplete polar will be tossed out
                print(f'{airfoil} set abandon!')
                continue

        # parameterize airfoil
        chebyshev_poly, _ = fs.fit_spline(f'Airfoil/{airfoil}.txt', deg=deg)
        # chebyshev_poly = camber_thickness(chebyshev_poly, deg)
        chebyshev_poly = chebyshev_poly.to_list()

        # sort cases in ascending order (alfa1 = -10 ... alfa31 = 20)
        cases = sorted(cases)

        # assemble aoa, cl, cd, cm, Xtop, Xbot polar data
        new_data = {}
        for i in range(len(cases)):
            new_data[f'alfa{i + 1}'] = [cases[i][0]]
            new_data[f'cl{i + 1}'] = [cases[i][1]]
            new_data[f'cd{i + 1}'] = [cases[i][2]]
            new_data[f'cm{i + 1}'] = [cases[i][3]]
            new_data[f'Xtop{i + 1}'] = [cases[i][4]]
            new_data[f'Xbot{i + 1}'] = [cases[i][5]]

        # Creating the reshaped DataFrame
        column_name = [f'a{j}' for j in range(1, 2*(deg+1)+2)]
        df1 = pd.DataFrame([[airfoil] + chebyshev_poly], columns=['airfoil name'] + column_name)
        df2 = pd.DataFrame(new_data)
        df = pd.concat([df, pd.concat([df1, df2], axis=1)], axis=0)

    # create panda dataframe and export to xlsx
    excel_name = f'M{mach}-Re{re:.2e}-N{ncrit}'.replace('.', '_')
    df.to_excel(f'Datasets/{excel_name}.xlsx', index=False, header=True)
    print(f'Angle of attacks recovered: {recovered}')
    return


def get_data_point(line):
    line_array = line.split()

    # extract alfa, cl, and cd
    data1 = line_array[0:3]

    # extract cm, Top_Xtr, and Bot_Xtr
    data2 = line_array[4::]

    # map variable from string to float
    data = list(map(float, data1+data2))
    return data


if __name__ == '__main__':
    t = time.time()
    pd.options.mode.chained_assignment = None  # default='warn'
    main()
    print(f'Total Time Elapsed: {time.time() - t:.4f}sec')
