import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def export_ice_absoprtion():
    # Data exported from publication Warren et al. 2006 (Fig. 1) but originated in
    # Perovich and Govoni 1991.
    # Data are interpolated to a fixed wavelength grid that fits with the wavelengths of
    # Seferian et al. 2018
    infile = "ice-absorption/sea_ice_absorption_perovich_and_govoni.csv"
    df = pd.read_csv(infile)
    print(df.head())
    # Define the grid to interpolate to
    wavelengths = np.arange(200, 1000, 10)
    k_ice = np.arange(0, 25, 0.01)
    # Get values from dataframe
    k_ice_pg = df["k_ice_pg"].values
    x_k_ice_pg = df["wavelength"].values
    # Do the interpolation
    interp_k_ice = np.interp(wavelengths, x_k_ice_pg, k_ice_pg)
    # Store data to csv file as dataframe
    data = {"wavelength": wavelengths, "k_ice_pg": interp_k_ice}
    df_out = pd.DataFrame(data, index=wavelengths)
    csv_filename = "ice-absorption/sea_ice_absorption_perovich_and_govoni_interpolated.csv"
    if os.path.exists(csv_filename): os.remove(csv_filename)
    df_out.to_csv(csv_filename, index=False)

    # Plot the result
    plt.plot(wavelengths, interp_k_ice, c="r", marker="o")
    plt.title("Absorption through sea-ice as  function of wavelength (Perovich and Govani 1991)")
    plt.show()


def export_chl_absorption():
    # Data exported from publication Matsuoka et al. 2007 (Table. 3)
    # Data are interpolated to a fixed wavelength grid that fits with the wavelengths of
    # Seferian et al. 2018
    infile = "chl-absorption/Matsuoka2007-chla_wavelength_absorption.csv"
    df = pd.read_csv(infile,sep=" ")
    print(df)

    # Get values from dataframe
    A_chl = df["A"].values
    B_chl = df["B"].values
    x_chl = df["wavelength"].values

    # Interpolate to 10 nm wavelength bands - only visible
    wavelengths = np.arange(400, 710, 10)

    # Do the interpolation
    A_chl_interp = np.interp(wavelengths, x_chl, A_chl)
    B_chl_interp = np.interp(wavelengths, x_chl, B_chl)

    # Create plots of absorption for all wavelengths from 400-700
    # for a range of chlorophyll values 0-10 mg/m-3
    chl = np.arange(0, 10, 0.1)
    all_a = np.empty((len(x_chl), len(chl)))
    ax1 = plt.subplot(2,1,1)
    for i_chl, c in enumerate(chl):
        for i_wave, x in enumerate(x_chl):

            all_a[i_wave, i_chl] = A_chl[i_wave] * c ** B_chl[i_wave]

        ax1.plot(x_chl, all_a[:, i_chl], c="k", alpha=0.5, linewidth=1)
        ax1.plot(x_chl, all_a[:, i_chl], c="r", marker="o",markersize=1)

    plt.title("Absorption a($\lambda$) for chl 0-10 mg/m$^{3}$ (Matsuoka et al. 2007)")
  #  plt.xlabel("Wavelength (nm)")
    plt.ylabel("absorption original (m$^{-1}$)")

    ax2 = plt.subplot(2, 1, 2)
    all_a = np.empty((len(wavelengths), len(chl)))
    for i_chl, c in enumerate(chl):
        for i_wave, x in enumerate(wavelengths):
            all_a[i_wave, i_chl] = A_chl_interp[i_wave] * c ** B_chl_interp[i_wave]

        ax2.plot(wavelengths, all_a[:, i_chl], c="k", alpha=0.5, linewidth=1)
        ax2.plot(wavelengths, all_a[:, i_chl], c="m", marker="o",markersize=1)

  #  plt.title("Absorption a($\lambda$) for chl 0-10 mg/m$^{3}$ (Matsuoka et al. 2007)")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("absorption interp (m$^{-1}$)")

    plt.show()




# df_out = pd.DataFrame(data, index=wavelengths)
# csv_filename = "ice-absorption/sea_ice_absorption_perovich_and_govoni_interpolated.csv"
# if os.path.exists(csv_filename): os.remove(csv_filename)
# df_out.to_csv(csv_filename, index=False)

# Plot the result

export_chl_absorption()
