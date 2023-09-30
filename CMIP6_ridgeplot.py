import calendar
import logging
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr

plt.style.use("mpl15")


class CMIP6_ridgeplot:
    @classmethod
    def calculate_climatology(
        cls,
        ds: xr.Dataset,
        infile: str,
        cumulative: bool = True,
        seasonal: bool = False,
    ):
        if cumulative:
            if seasonal:
                climatology = ds.groupby("time.season").sum("time", keep_attrs=True)
            else:
                climatology = ds.groupby("time.month").sum("time", keep_attrs=True)
        else:
            # Calculate the climatology and the anomalies from the de-trended dataset
            if seasonal:
                climatology = ds.groupby("time.season").mean(
                    "time", keep_attrs=True, skipna=True
                )
            else:
                climatology = ds.groupby("time.month").mean(
                    "time", keep_attrs=True, skipna=True
                )

        logging.info(f"[CMIP6_ridgeplot] Finished calculating climatology for {infile}")
        return climatology

    # Define and use a simple function to label the plot in axes coordinates
    @classmethod
    def label(cls, x, color, label):
        ax = plt.gca()
        months = [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ]
        # months[int(label) - 1]

        ax.text(
            0,
            0.2,
            label,
            fontweight="regular",
            color="black",
            ha="left",
            va="center",
            transform=ax.transAxes,
        )

    @classmethod
    def return_df_climatology(
        cls,
        var_name: str,
        infile: str,
        start_time: datetime,
        end_time: datetime,
        depth_threshold: float,
        ds: xr.Dataset = None,
        cumulative: bool = True,
        seasonal: bool = False,
    ) -> pd.DataFrame:
        """
        Create a Pandas dataframe organized as monthly climatology for a specific period
        of time from a dataset.
         If ds is not Noe, we pass a dataset instead of file to the function. The filename is then ignored and
         we use the dataset instead.
        :param depth_threshold: If provides will filter dataset to include only depths less than depth_threshold
        :param infile: string
        :param start_time: datetime
        :param end_time: datetime
        :param ds: xr.Dataset
        :param cumulative: bool
        :param seasonal: bool
        :return: pd.DataFrame

        """
        if ds is None:
            ds = xr.open_dataset(infile).sel(time=slice(start_time, end_time))
            print(f"Opening file {infile}")
        else:
            ds = ds.sel(time=slice(start_time, end_time))

        ds = ds.chunk({"time": -1})
        if depth_threshold is not None:
            if "depth" in ds.variables:
                ds = ds.where(ds.depth < depth_threshold)
            elif "depth_mean" in ds.variables:
                ds = ds.where(ds.depth_mean < depth_threshold)

        clim = cls.calculate_climatology(ds, infile, cumulative, seasonal)

        return clim.to_dataframe()

    @classmethod
    def indicate_maximum(cls, color, g1, labels, ind, panels):
        all_axes = g1.axes.ravel()
        g1.map(plt.axhline, y=0, color="black", lw=0.5, clip_on=False)

        # Loop over all the axes (equal to panels either 4 or 12),
        # extract the data that draws the polyline (the KDE) for each period
        # and find maximum values and add a dot at that
        res = {}
        for ax_i, ax in enumerate(all_axes):
            ax.set_ylabel("")
            label_text = [label for label in labels]
            if ax_i == 0 and ind == 0:
                ax.legend(
                    labels=label_text,
                    facecolor="white",
                    framealpha=1,
                    loc="upper right",
                )
            counter = 0
            for i in ax.get_children():
                if i.__class__.__name__ == "PolyCollection":
                    if ind == (counter % panels):
                        x, y = i.get_paths()[0].vertices.T

                        maxid = y.argmax()
                        ax.plot(x[maxid], y[maxid], marker="o", color=color, ms=10)

                        # The following stores the maximum/cumulative value per month as a dataframe
                        # to be used afterwards for plotting the change between time periods and months.
                        # This is to see if one month now vs in the future will go up or down in productivity
                        # FRA to SFO, 26.10.2022

                        if res:
                            xx = res["x"]
                            yy = res["y"]
                            xx.append(x[maxid])
                            yy.append(y[maxid])
                        else:
                            xx = [x[maxid]]
                            yy = [y[maxid]]

                        res["x"] = xx
                        res["y"] = yy
                    counter += 1

        return pd.DataFrame(data=res)

    @classmethod
    def create_ridgeplot(
        cls,
        var_name,
        df: pd.DataFrame,
        ds: xr.Dataset,
        outfile: str,
        labels: str,
        seasonal: bool = True,
        cumulative: bool = False,
    ) -> None:
        logging.info(
            "[CMIP6_ridgeplot] Starting ridgeplot creation for {}".format(var_name)
        )

        sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)}, font_scale=3.4)

        # Color combinations
        # https://digitalsynopsis.com/design/minimal-web-color-palettes-combination-hex-code/
        color1 = "#94c492"
        color2 = "#34868e"
        color3 = "#323d77"
        color4 = "#283d77"
        color5 = "#203d77"
        color6 = "#153d77"

        color1 = "#8172B3"
        color2 = "#64B5CD"

        colors = [
            color1,
            color2,
            color3,
            color4,
            color5,
            color6,
            color1,
            color2,
            color3,
            color4,
            color5,
            color6,
            color1,
            color2,
            color3,
            color4,
            color5,
            color6,
            color1,
            color2,
            color3,
            color4,
            color5,
            color6,
            color1,
            color2,
            color3,
            color4,
            color5,
            color6,
            color1,
            color2,
            color3,
            color4,
            color5,
            color6,
        ]

        max_df_dict = {}
        clip = None

        if seasonal:
            panels = 4
            g1 = sns.FacetGrid(df, row="season", hue="season", aspect=15, height=1.2)
        else:
            panels = 12
            g1 = sns.FacetGrid(df, row="month", hue="month", aspect=15, height=1.2)

        cumulative_plot = False
        if cumulative:
            log = False
        else:
            log = False

        for i, label in enumerate(labels):
            # Draw the densities in a few steps
            g1.map(
                sns.kdeplot,
                label,
                bw_adjust=0.5,
                cumulative=cumulative_plot,
                clip_on=True,
                fill=True,
                multiple="layer",
                alpha=0.7,
                linewidth=1.5,
                color=colors[i],
                log_scale=log,
                legend=True,
                clip=clip,
            )
            g1.map(
                sns.kdeplot,
                var_name,
                clip_on=False,
                multiple="layer",
                cumulative=cumulative_plot,
                color=colors[i],
                lw=1,
                bw_adjust=0.5,
                log_scale=log,
                legend=False,
                clip=clip,
            )

            max_df_dict[label] = cls.indicate_maximum(colors[i], g1, labels, i, panels)

        if not cumulative:
            # Define and use a simple function to label the plot in axes coordinates
            g1.map(cls.label, var_name)
            # Set the subplots to overlap
            g1.fig.subplots_adjust(hspace=-0.25)

            # Remove axes details that don't play well with overlap
            g1.set_titles("")
            g1.set(yticks=[])

            g1.despine(top=True, left=True)
            all_axes = g1.axes.ravel()
            for ax_i, ax in enumerate(all_axes):
                ax.set_ylabel("")
                if ax_i == 0:
                    ax.legend(
                        labels=[labels[0], labels[1]],  # , labels[2]],
                        facecolor="white",
                        framealpha=1,
                        loc="upper right",
                    )

            if not os.path.exists(os.path.dirname(outfile)):
                os.makedirs(os.path.dirname(outfile))
            if os.path.exists(outfile):
                os.remove(outfile)
            print("[CMIP6_plot] Created plot {}".format(outfile))
            plt.savefig(outfile, dpi=300)

            plt.show()

        if cumulative:
            plt.subplots(figsize=(10, 8))
            cls.heatmap_of_change(max_df_dict, var_name, labels, panels)
            plt.show()

    @classmethod
    def heatmap_of_change(
        self,
        max_df_dict: [],
        var_name: str,
        labels: [],
        panels: [],
        LME: str,
        scenario: str,
    ):
        clim = np.empty((panels, len(labels)))

        for i, label in enumerate(labels):
            #      max_df_dict[label]=(max_df_dict[label]-max_df_dict[label].mean())/max_df_dict[label].std()
            if i == 0:
                df = max_df_dict[label]  # - max_df_dict[label]
                clim[:, i] = df.values
            else:
                # Calculate the relative change for a period relative to start period (1993-2020)
                df = max_df_dict[label]
                if var_name not in ["uvb_mean", "par_mean", "uv_mean", "uva_mean"]:
                    clim[:, i] = np.squeeze(np.array(df.values)) - clim[:, 0]
                else:
                    clim[:, i] = (
                        (np.squeeze(np.array(df.values)) - clim[:, 0])
                        / clim[:, 0]
                        * 100.0
                    )

        clim = np.where(abs(clim) > 1e3, np.nan, clim)
        month_labels = [
            calendar.month_name[i] for i in range(1, 13)
        ]  # labels for y-axis

        def convert_to_string(i: float):
            if abs(np.around(i, 1)) > 0.00000005:
                return f"{np.around(i,1)}"
            return ""

        def convert_uvb_to_string(i: float):
            if abs(np.around(i, 3)) > 0.00000005:
                return f"{np.around(i,2)}"
            return ""

        applyall = np.vectorize(convert_to_string)
        annot_labels = applyall(clim)
        fontsize = 10
        fmt = ""
        vmin = np.min(clim)
        vmax = np.max(clim)

        if var_name == "chl_mean":
            cmap = sns.color_palette("PiYG", as_cmap=True)
            applyall = np.vectorize(convert_to_string)
            vmin = -0.8
            vmax = 0.8
        elif var_name in ["par_mean"]:
            vmin = 0
            vmax = 100
            annot_labels = applyall(clim)
            fontsize = 7
            cmap = sns.color_palette("rocket_r", as_cmap=True)
        elif var_name in ["tos_mean"]:
            vmin = -1
            vmax = 8
            annot_labels = applyall(clim)
            fontsize = 7
            cmap = sns.color_palette("rocket_r", as_cmap=True)
        elif var_name in ["sisnthick_mean"]:
            applyall = np.vectorize(convert_uvb_to_string)
            vmin = 0
            vmax = 0.2
            annot_labels = applyall(clim)
            fontsize = 6
            cmap = sns.color_palette("Blues", as_cmap=True)
        elif var_name in ["sithick_mean"]:
            vmin = -3
            vmax = 0
            annot_labels = applyall(clim)
            fontsize = 7
            cmap = sns.color_palette("Blues", as_cmap=True)
        elif var_name in ["tas_mean"]:
            vmin = -15
            vmax = 15
            annot_labels = applyall(clim)
            fontsize = 7
            cmap = sns.color_palette("PiYG", as_cmap=True)
        elif var_name in ["uvb_mean"]:
            applyall = np.vectorize(convert_to_string)
            annot_labels = applyall(clim)
            fontsize = 7
            vmin = -50
            vmax = 50
            cmap = sns.color_palette("PiYG", as_cmap=True)
        elif var_name in ["uvi_mean"]:
            applyall = np.vectorize(convert_to_string)
            annot_labels = applyall(clim)
            fontsize = 7
            vmin = -0.5
            vmax = 0.5
            cmap = sns.color_palette("PiYG", as_cmap=True)
        elif var_name in ["uv_mean"]:
            applyall = np.vectorize(convert_to_string)
            annot_labels = applyall(clim)
            fontsize = 7
            vmin = 0
            vmax = 100
            cmap = sns.color_palette("rocket_r", as_cmap=True)
        elif var_name in ["clt_mean"]:
            applyall = np.vectorize(convert_to_string)
            annot_labels = applyall(clim)
            fontsize = 7
            vmin = -5
            vmax = 5
            cmap = sns.color_palette("PiYG", as_cmap=True)
        elif var_name in ["siconc_mean"]:
            applyall = np.vectorize(convert_to_string)
            annot_labels = applyall(clim)
            fontsize = 7
            vmin = -100
            vmax = 0
            cmap = sns.color_palette("Blues", as_cmap=True)
        elif var_name in ["uva_mean"]:
            applyall = np.vectorize(convert_to_string)
            annot_labels = applyall(clim)
            fontsize = 7
            vmin = -10
            vmax = 10
            cmap = sns.color_palette("rocket_r", as_cmap=True)
        else:
            cmap = sns.color_palette("rocket_r", as_cmap=True)
        with sns.axes_style("white"):
            sns.heatmap(
                clim,
                square=True,
                cmap=cmap,
                xticklabels=labels,
                vmin=vmin,
                vmax=vmax,
                yticklabels=month_labels,
                annot=annot_labels,
                fmt=fmt,
                linewidth=0.5,
                annot_kws={"fontsize": fontsize},
            )

            outfile = f"Figures/heatmap_{var_name}_ensemble_{LME}_{scenario}.png"
            plt.savefig(outfile, dpi=300, bbox_inches="tight", pad_inches=0)

    @classmethod
    def ridgeplot(
        cls,
        var_name,
        infile,
        outfile,
        glorys=False,
        depth_threshold=None,
        ds: xr.Dataset = None,
        cumulative: bool = False,
        seasonal: bool = False,
    ):
        if not os.path.exists(os.path.dirname(outfile)):
            os.makedirs(os.path.dirname(outfile), exist_ok=True)

        start_times = [datetime(1993, 1, 1), datetime(2012, 1, 1)]
        end_times = [datetime(2012, 1, 1), datetime(2020, 1, 1)]

        step = 10
        # TODO: Add longer historical period by joining two lists.
        #  start_times.extend([datetime(2020+ i, 1, 1) for i in range(0,70, step)])
        #  end_times.extend([datetime(2020 + step + i, 1, 1) for i in range(0, 70, step)])

        #  start_times = [datetime(2020, 1, 1), datetime(2090, 1, 1)]
        #  end_times = [datetime(2030, 1, 1), datetime(2099, 12, 1)]

        labels = []
        dfs = []
        if seasonal:
            frequency = "season"
        else:
            frequency = "month"

        for start, end in zip(start_times, end_times):
            labels.append(
                f"{start.year}-{end.year}",
            )

            df = cls.return_df_climatology(
                var_name,
                infile,
                start_time=start,
                end_time=end,
                depth_threshold=depth_threshold,
                ds=ds,
                cumulative=cumulative,
                seasonal=seasonal,
            ).reset_index(level=frequency)
            dfs.append(df)

            print(f"Created dataframe for {start.year}-{end.year}: {df.describe()}")

        for i, df in enumerate(dfs):
            start = start_times[i]
            end = end_times[i]
            print(f"{start.year}-{end.year}")
            if i == 0:
                df_combined = df
                df_combined[f"{start.year}-{end.year}"] = df[var_name]
            else:
                df_combined[f"{start.year}-{end.year}"] = df[var_name]

        cls.create_ridgeplot(
            var_name, df_combined, ds, outfile, labels, seasonal, cumulative
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info("[CMIP6_ridgeplot] Initialized logging")

    cumulative = True
    seasonal = True

    infile = "../shared/cmip6/downscaling/NewFoundland/ssp585/ensemble/thetao/thetao_ensemble_sd+ba_surface_depth_5_stats_ssp585.nc"
    outfile = "Figures/thetao_ensemble_sd+ba_surface_depth_5_stats_ssp585_300m.png"

    CMIP6_ridgeplot.ridgeplot(
        "thetao_mean",
        infile,
        outfile,
        glorys=False,
        depth_threshold=9000,
        cumulative=cumulative,
        seasonal=seasonal,
    )
