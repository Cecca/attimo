import marimo

__generated_with = "0.13.6"
app = marimo.App(layout_file="layouts/vizdata.grid.json")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import os
    from pathlib import Path
    import numpy as np
    from scipy.io import loadmat
    return loadmat, mo, np, plt


@app.cell
def _(mo):
    fbrowser = mo.ui.file_browser("data", multiple=False)
    fbrowser
    return (fbrowser,)


@app.cell
def _(fbrowser, loadmat):
    def loaddata(path):
        data = loadmat(path)
        for name in data.keys():
            if not hasattr(data[name], "shape"):
               continue
            return data[name].flatten()


    ts = loaddata(fbrowser.value[0].path)
    ts
    return (ts,)


@app.cell
def _(mo):
    asel = mo.ui.number(label="a", value=0)
    bsel = mo.ui.number(label="b", value=0)
    wsel = mo.ui.number(label="window", value=128)
    mo.vstack((asel, bsel, wsel))
    return asel, bsel, wsel


@app.cell
def _(asel, bsel, plt, ts):
    plt.plot(ts)   
    plt.axvline(asel.value, color="red")
    plt.axvline(bsel.value, color="red")
    plt.title("Subsequences in context")
    return


@app.cell
def _(asel, bsel, np, plt, ts, wsel):
    a = ts[asel.value:asel.value+wsel.value]
    b = ts[bsel.value:bsel.value+wsel.value]
    plt.plot(znorm(a))
    plt.plot(znorm(b))
    plt.title("distance: " + str(np.linalg.norm(znorm(a) - znorm(b))))
    return


@app.cell
def _():
    return


@app.function
def znorm(sub):
    return (sub - sub.mean()) / sub.std()


if __name__ == "__main__":
    app.run()
