import marimo

__generated_with = "0.13.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import TABLEAU_COLORS
    from scipy.stats import norm
    import marimo as mo
    return TABLEAU_COLORS, mo, norm, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""$||x - y||^2_2 = ||x||^2 + ||y||^2 - 2\langle x, y\rangle$""")
    return


@app.cell
def _(np):
    w = 1024
    dists = np.linspace(0, 45, 1000)
    return dists, w


@app.cell
def _(np):
    def simhash(eucls, w):
        """Compute the simhash collision probability of points given their _Euclidean_ distance"""
        dots = (2*w - eucls*eucls)/2
        theta = np.acos(dots / w)
        return 1 - theta / np.pi
    return (simhash,)


@app.cell
def _(norm, np):
    def rproj(d, r):
        """Compute the E2LSH collision probability of points given their Euclidean distance and quantization width `r`"""
        return (
            1.0
            - 2.0 * norm.cdf(-r / d)
            - (2.0 / (np.sqrt(np.pi * 2.0) * (r / d)))
            * (1.0 - np.exp(-r * r / (2.0 * d * d)) )
        )
    return (rproj,)


@app.cell
def _():
    Krproj = 16
    Ksimhash = 128
    L = 128
    return Krproj, Ksimhash, L


@app.cell
def _(Krproj, Ksimhash):
    Ksimhash / Krproj
    return


@app.cell
def _(L, TABLEAU_COLORS, dists, plt, rproj):
    # for k in range(64, 129, 8):
    #     plt.plot(dists, 1 - (1 - simhash(dists, w)**(k))**L, label=f"simhash(k={k})", linestyle="dotted")

    for k, color in zip(range(4, 17, 2), TABLEAU_COLORS.keys()):
        for qw, lstyle in zip([4.36, 0.7], ["solid", "dashed", "dotted"]):
            success_probs = 1 - (1 - rproj(dists, qw)**(k))**L
            print(f"k={k} qw={qw} largest confirmed: ", dists[success_probs >= 0.9][-1])
            plt.plot(dists, success_probs, c=color, label=f"rproj(k={k}, qw={qw})", linestyle=lstyle)

    
    # plt.axhline(0.9, c="black")
    plt.semilogy()
    plt.legend(bbox_to_anchor=(1.1, 1.05))
    return


@app.cell
def _(dists, np, plt, simhash, w):
    def simhash_rho(r1=1.0):
        rdists = dists[dists > r1]
        cs = rdists / r1
        rho = np.log(1/simhash(r1, w)) / np.log(1/simhash(rdists, w))

        plt.plot(cs, rho)
        plt.plot(cs, 1/cs, label="reference", linestyle="dotted")
        plt.semilogy
        plt.show()

    simhash_rho(1)
    return


@app.cell
def _(dists, np, plt, rproj):
    def rproj_rho(qw=4.36):
        r1 = 1.0
        rdists = dists[dists > r1]
        cs = rdists / r1
        rho = np.log(1/rproj(r1, qw)) / np.log(1/rproj(rdists, qw))

        plt.plot(cs, rho)
        plt.plot(cs, 1/cs, label="reference", linestyle="dotted")
        plt.semilogy()
        plt.show()

    rproj_rho(qw=20)
    return


if __name__ == "__main__":
    app.run()
