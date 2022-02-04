# coding: utf-8
#
# This script visualizes the solution of the time-fractal heat equation
#     \partial_t^\alpha u (t,x) = \nabla D \nabla u (t,x)
#     u(0,x) = sin( \pi x)
#     u(t,0) = u(t,1) = 0 \forall t
#     t \in [0,1], x \in [0,1]
#
# The analytic solution u is given by the Mittag-Leffler Function as
#     u(t,x) = E_{\alpha,1.0}(- \pi^2 t^\alpha) sin(\pi x)
# The script accepts various parameter to steer the computation and plot u(t,x) 
# and E_{\alpha,1.0}(- \pi^2 t^\alpha).
#
# Set up conda environment to run the script:
#   `conda env create -f py_scripts/conda_environment.yml`
#   `conda list | grep fractional`
#   `conda activate fractional`
#
# Get information on how to use the script:
#   `python py_scripts/visualize_ML.py --help`
#
# Run the script
#   `python py_scripts/visualize_ML.py --alpha 0.5`
#
# Author: Tobias Duswald
#


from absl import app
from absl import flags
import numpy as np
from matplotlib import pyplot as plt
import sys, os

sys.path.append(os.getcwd() + "/mittag_leffler")
import mittag_leffler as ml


FLAGS = flags.FLAGS
flags.DEFINE_float(
    "alpha", 0.5, r"Exponent alpha in \Gamma_{alpha,beta}(-pi^2*t^alpha)"
)
flags.DEFINE_float("beta", 1.0, "Lower bound considered in approximation.")
flags.DEFINE_float("tmax", 1, "Upper bound.")
flags.DEFINE_float("tmin", 0, "Lower bound.")
flags.DEFINE_integer("npoints", 100, "Number of points for sampling.")
flags.DEFINE_boolean(
    "visualize", False, "Visualize the convergence of the approximation."
)


def main(argv):
    """
    Main function. Plots the ML (Mittag-Leffler Function)
    """

    # Points to evaluate function on
    t = np.linspace(FLAGS.tmin, FLAGS.tmax, FLAGS.npoints)
    ft = -np.pi**2 * np.power(t, FLAGS.alpha)

    # Define the ML function
    ml_func = np.vectorize(ml.ml)
    function_values = ml_func(ft, FLAGS.alpha, FLAGS.beta)

    # Compute 1 D solution at different times
    x = np.linspace(0, 1, 200)
    n = int(FLAGS.npoints / 5)
    solutions = []
    for i in range(5):
        a = np.sin(np.pi * x) * function_values[i * n]
        solutions.append(a)
    solutions.append(np.sin(np.pi * x) * function_values[-1])

    # Visualize approximation
    fig, axs = plt.subplots(2)
    fig.suptitle(
        r"Solution: $\alpha={}, \beta={}$".format(FLAGS.alpha, FLAGS.beta)
    )

    axs[0].plot(
        t, function_values, label=r"$E_{\alpha,\beta}(-\pi^2 t^{\alpha})$"
    )
    axs[0].legend(loc="upper right")
    axs[0].set_ylabel("f(t)")
    axs[0].set_xlabel("t")
    axs[0].grid(True)

    for i in range(6):
        axs[1].plot(x, solutions[i], label="ts {}/5".format(i))
    axs[1].legend(loc="upper right")
    axs[1].set_ylabel(r"$f_t(x)$")
    axs[1].set_xlabel("x")
    axs[1].grid(True)

    plt.show()


if __name__ == "__main__":
    app.run(main)
