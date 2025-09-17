""" Simple script to test the Z-tool for a single-bus analysis """

# --- Fix for local imports ---
import sys
import numpy as np
import matplotlib.pyplot as plt
from os import getcwd, path, makedirs

# Make sure to update this path to the location of your 'Source' folder
# This assumes the 'ztoolacdc' library is inside the 'Source' folder
sys.path.insert(0, r"C:\Users\amanm\Desktop\Electrical Core\Bode Plot\Z-tool-main\Source") 

from ztoolacdc import *

""" -------------------- Bode Plot and Data Export Function ---------------------- """

def plot_admittance_bode(admittance_obj, save_folder, file_name_prefix):
    """
    Generates and saves a Bode plot and its corresponding data as a text file.

    The function plots the magnitude (in dB) and phase (in degrees) for all four
    elements of the admittance matrix (Ydd, Ydq, Yqd, Yqq). It also saves this
    frequency response data to a .txt file for further analysis.
    """
    if admittance_obj.y.shape[1:] != (2, 2):
        print(f"Warning: Bode plot function is designed for 2x2 matrices. "
              f"Skipping plot for {file_name_prefix} with shape {admittance_obj.y.shape[1:]}.")
        return

    # --- Data Extraction and Calculation ---
    frequencies = admittance_obj.f
    Y = admittance_obj.y
    elements = [(0, 0), (0, 1), (1, 0), (1, 1)]
    
    # Prepare lists to hold all calculated data for text file export
    magnitudes_db = []
    phases_deg = []

    for row, col in elements:
        magnitudes_db.append(20 * np.log10(np.abs(Y[:, row, col])))
        phases_deg.append(np.angle(Y[:, row, col], deg=True))

    # --- 1. Save Frequency Response Data to Text File (NEW FEATURE) ---
    # Combine all data into a single array for saving
    # Columns: Freq, Mag_dd, Mag_dq, Mag_qd, Mag_qq, Phase_dd, Phase_dq, Phase_qd, Phase_qq
    data_to_save = np.stack([frequencies] + magnitudes_db + phases_deg, axis=1)
    
    # Define the header for the text file
    header = ("Frequency(Hz)\t"
              "Mag_Ydd(dB)\tMag_Ydq(dB)\tMag_Yqd(dB)\tMag_Yqq(dB)\t"
              "Phase_Ydd(deg)\tPhase_Ydq(deg)\tPhase_Yqd(deg)\tPhase_Yqq(deg)")
              
    # Define the output path and save the file
    data_output_path = path.join(save_folder, f"{file_name_prefix}_Frequency_Response.txt")
    np.savetxt(data_output_path, data_to_save, delimiter='\t', header=header, comments='')
    print(f"Frequency response data saved to: {data_output_path}")

    # --- 2. Generate and Save Bode Plot PDF ---
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), sharex=True)

    labels = [r'$Y_{dd}$', r'$Y_{dq}$', r'$Y_{qd}$', r'$Y_{qq}$']
    styles = ['-', '--', '-.', ':']
    colors = ['b', 'r', 'g', 'm']

    for i in range(len(elements)):
        ax[0].plot(frequencies, magnitudes_db[i], label=labels[i], linestyle=styles[i], color=colors[i])
        ax[1].plot(frequencies, phases_deg[i], label=labels[i], linestyle=styles[i], color=colors[i])

    ax[0].set_ylabel('Magnitude [dB]')
    ax[0].set_xscale("log")
    ax[0].grid(True, which='both', linestyle='--', linewidth=0.5)
    ax[0].set_title(f'Bode Plot of Admittance: {file_name_prefix}')
    ax[0].legend()

    ax[1].set_xlabel('Frequency [Hz]')
    ax[1].set_ylabel('Phase [°]')
    ax[1].set_xscale("log")
    ax[1].set_yticks([-180, -90, 0, 90, 180])
    ax[1].grid(True, which='both', linestyle='--', linewidth=0.5)
    ax[1].legend()

    plt.tight_layout()
    plot_output_path = path.join(save_folder, f"{file_name_prefix}_Bode_Plot.pdf")
    fig.savefig(plot_output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Bode plot saved to: {plot_output_path}")


""" -------------------- PSCAD PROJECT ---------------------- """
pscad_folder = r"C:\Users\amanm\Desktop\Electrical Core\Bode Plot\Z-tool-main\Examples\2L_VSC\\"  
workspace_name = "sample"

results_folder = path.join(getcwd(), "results")
if not path.exists(results_folder):
    makedirs(results_folder)

project_name = "Simple_2L_VSC_RLC"

""" -------------------- SCAN SETTINGS ---------------------- """
f_points = 8 * 50
f_base = 0.5
f_min = 1.0
f_max = 500.0

start_fft = 1.0
fft_periods = 1
dt_injections = 1

t_snap = 10
t_sim = start_fft + fft_periods / f_base
t_step = 20.0
v_perturb_mag = 0.02

output_files = 'single_bus_example'

""" -------------------- Frequency scan ---------------------- """
freq = create_freq.loglist(f_min=f_min, f_max=f_max, f_points=f_points, f_base=f_base)

frequency_sweep.frequency_sweep(
    t_snap=t_snap,
    t_sim=t_sim,
    t_step=t_step,
    dt_injections=dt_injections,
    f_base=f_base,
    freq=freq,
    start_fft=start_fft,
    fft_periods=fft_periods,
    v_perturb_mag=v_perturb_mag,
    working_dir=pscad_folder,
    workspace_name=workspace_name,
    project_name=project_name,
    results_folder=results_folder,
    output_files=output_files,
    show_powerflow=True,
    component_parameters=[["pq_tau_meas", 0.001]]
)

""" -------------------- Admittance Reading ---------------------- """
Y_VSC = read_admittance.read_admittance(path=results_folder, involved_blocks=["PCC-1"], file_root=output_files)
Y_grid = read_admittance.read_admittance(path=results_folder, involved_blocks=["PCC-2"], file_root=output_files)

""" -------------------- Bode Plot Visualization & Data Export ---------------------- """
bode_plot_folder = path.join(results_folder, "Bode_Plots")
if not path.exists(bode_plot_folder):
    makedirs(bode_plot_folder)

print("\nCreating Bode plots and exporting frequency response data...")
plot_admittance_bode(Y_VSC, save_folder=bode_plot_folder, file_name_prefix="Y_VSC")
plot_admittance_bode(Y_grid, save_folder=bode_plot_folder, file_name_prefix="Y_Grid")

""" -------------------- Stability analysis ---------------------- """
print("\nAnalysis of the PSCAD case")

L = np.matmul(np.linalg.inv(Y_grid.y), Y_VSC.y)
stability.nyquist(L, Y_VSC.f, results_folder=results_folder, filename="PSCAD_case")
stability.EVD(G=np.linalg.inv(Y_grid.y + Y_VSC.y), frequencies=Y_VSC.f, results_folder=results_folder, filename="PSCAD_case")
stability.passivity(G=Y_VSC.y, frequencies=Y_VSC.f, results_folder=results_folder, filename="PSCAD_case", Yedge=Y_grid.y)
stability.small_gain(G1=np.linalg.inv(Y_grid.y), G2=Y_VSC.y, frequencies=Y_VSC.f, results_folder=results_folder, filename="PSCAD_case")

""" -------------------- Stability analysis with different series compensation values ---------------------- """
print("\nAnalysis of different series compensation values using previously scanned converter admittance")

Wpu = np.array([[0, 1], [-1, 0]])
w0 = 2 * np.pi * 50
Z_RL = np.linalg.inv(Y_grid.y)
X_g = np.real(Z_RL[1, 0, 1])
print("The grid-side inductance is", X_g / (2 * np.pi * 50), "H")

comp_level = np.arange(0.05, 0.7, 0.01)
Y_C = np.empty((len(Y_grid.f), 2, 2), dtype='cdouble')
stability_assessment = []

# Create a dedicated subfolder for the compensation analysis results
compensation_folder = path.join(results_folder, "Compensation")
if not path.exists(compensation_folder):
    makedirs(compensation_folder)

for case in range(len(comp_level)):
    C_g = 1 / (w0 * comp_level[case] * X_g)
    for f_point, f in enumerate(Y_grid.f):
        Y_C[f_point, ...] = 1j * 2 * np.pi * f * C_g * np.identity(2) + w0 * C_g * Wpu
    Z_RLC = np.linalg.inv(Y_C) + Z_RL
    stable = stability.nyquist(np.matmul(Z_RLC, Y_VSC.y), Y_VSC.f,
                                 results_folder=compensation_folder,
                                 filename="PSCAD_case_" + str(case), verbose=False)
    stability.nyquist_det(L=np.matmul(Z_RLC, Y_VSC.y), frequencies=Y_VSC.f,
                            results_folder=compensation_folder,
                            filename="PSCAD_case_" + str(case))
    stability_assessment.append(stable)
    if not stable:
        print(round(comp_level[case] * 100, 1), "% series compensation is small-signal unstable")
        stability.EVD(G=np.linalg.inv(np.linalg.inv(Z_RLC) + Y_VSC.y), frequencies=Y_VSC.f,
                      results_folder=compensation_folder,
                      filename="PSCAD_case_" + str(case), verbose=False)
        stability.nyquist_det(L=np.matmul(Z_RLC, Y_VSC.y), frequencies=Y_VSC.f,
                                results_folder=compensation_folder,
                                filename="PSCAD_case_" + str(case))

print("\nCompensation larger than", round(comp_level[sum(stability_assessment)] * 100, 1), "% might result in instability")