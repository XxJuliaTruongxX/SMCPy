import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import time


def eval_model(theta):
    time.sleep(0.05)  # artificial slowdown to show off progress bar
    a = theta[:, 0, None]
    b = theta[:, 1, None]
    return a * np.arange(45, 61) + b


def generate_data(x_true, eval_model, std_dev, plot=True):
    y_true = eval_model(x_true)
    noisy_data = y_true + np.random.normal(0, std_dev, y_true.shape)
    if plot:
        plot_noisy_data(x, y_true, noisy_data)
    return noisy_data


def plot_noisy_data(x, y_true, noisy_data):
    fig, ax = plt.subplots(1)
    ax.plot(x.flatten(), y_true.flatten(), "-k")
    ax.plot(x.flatten(), noisy_data.flatten(), "o")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.show()


def plot_posterior(step_list, out_file_base, show_mcmc=False, figsize=None):
    for param_idx, param in enumerate(step_list[-1].param_dict.keys()):
        title = (
            "Without Proposal: " if "without" in out_file_base else "With Proposal: "
        )
        out_file = out_file_base + f"_{param}.png"

        phi_seq = np.array([])
        for phi in step_list.phi_sequence:
            phi_lst = ["%.3f" % (phi)] * step_list[-1].num_particles
            phi_seq = np.hstack((phi_seq, phi_lst))

        param_vals = np.array([])
        for step in step_list:
            param_vals = np.hstack((param_vals, step.params[:, param_idx]))

        data = pd.DataFrame({"phi": phi_seq, title + param: param_vals})

        sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
        pal = sns.color_palette("crest", n_colors=len(data.phi.unique()))
        g = sns.FacetGrid(
            data,
            row="phi",
            hue="phi",
            aspect=15,
            height=0.5,
            palette=pal,
            sharex=True,
            sharey=False,
        )

        def custom_kdeplot(data, color, **kwargs):
            bw = 0.5  # Default bandwidth
            if data["phi"].iloc[0] == "MCMC":  # Adjust for last facet
                bw = 2.5  # Change bw_adjust for last facet
            sns.kdeplot(
                data=data,
                x=title + param,
                bw_adjust=bw,
                clip_on=False,
                fill=True,
                alpha=1,
                linewidth=1.5,
                color=color,
            )
            sns.kdeplot(
                data=data, x=title + param, bw_adjust=bw, clip_on=False, color="w", lw=2
            )

        g.map_dataframe(custom_kdeplot)

        g.refline(y=0, linewidth=1, linestyle="-", color=None, clip_on=False)

        def label(x, color, label):
            # if label == "0.034":
            label = f"$\phi = {label}$" if label != "MCMC" else label
            ax = plt.gca()
            ax.text(
                0,
                0.25,
                label,
                # fontweight="bold",
                fontsize=10,
                color=color,
                ha="left",
                va="center",
                transform=ax.transAxes,
            )

        g.map(label, title + param)

        g.figure.subplots_adjust(hspace=-0.25)
        g.set_titles("")
        g.set(yticks=[], ylabel="")
        g.despine(bottom=True, left=True)
        plt.xlabel(title + param)

        if show_mcmc:
            ax = g.axes[-1, 0]
            ax.set_position(
                [ax.get_position().x0, ax.get_position().y0 - 0.1, 0.775, 0.2]
            )
            g.fig.set_size_inches(g.fig.get_size_inches())

        if figsize is not None:
            g.figure.set_size_inches(*figsize)

        plt.savefig(
            "SMCPy/examples/proposal_example/" + out_file,
            bbox_inches="tight",
            pad_inches=0,
        )
