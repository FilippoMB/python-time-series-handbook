import matplotlib.pyplot as plt

def run_sequence_plot(x, y, title, xlabel="Time", ylabel="Values", ax=None):
    """
    Plot the time series data

    Parameters
    ----------
    x : array-like
        The time values.
    y : array-like
        The values of the time series.
    title : str
        The title of the plot.
    xlabel : str
        The label for the x-axis.
    ylabel : str
        The label for the y-axis.
    ax : matplotlib axes
        The axes to plot on.

    Returns
    -------
    ax : matplotlib axes
        The axes object with the plot.
    """
    if ax is None:
        _, ax = plt.subplots(1,1, figsize=(10, 3.5))
    ax.plot(x, y, 'k-')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    return ax