"""Small SciPy example for matched peak-gain optimization at 868 MHz."""

from scipy.optimize import differential_evolution

from emerge_loaded_antenna import (
    AntennaDesign,
    DesignSpace,
    DesignVariable,
    GainMatchObjective,
)


space = DesignSpace(
    AntennaDesign(),
    (
        DesignVariable("bottom_length", 100e-3, 180e-3),
        DesignVariable("middle_length", 160e-3, 260e-3),
        DesignVariable("top_length", 100e-3, 180e-3),
        DesignVariable("coil1.pitch", 5e-3, 10e-3),
        DesignVariable("coil2.pitch", 5e-3, 10e-3),
        DesignVariable("radial_length", 60e-3, 100e-3),
    ),
)

# Minimize mismatch penalty - peak gain. Designs below -10 dB S11 receive no
# additional mismatch penalty, so the optimizer then concentrates on gain.
objective = GainMatchObjective(
    space,
    target_frequency=868e6,
    maximum_s11_db=-10.0,
    mismatch_weight=2.0,
    gain_weight=1.0,
)

result = differential_evolution(
    objective,
    bounds=space.bounds,
    maxiter=20,
    popsize=8,
    polish=False,
    seed=1,
    workers=1,  # EMerge/Gmsh model state is process-global, not thread-safe.
    updating="immediate",
)

best_design = space.decode(result.x)
print("Best objective:", result.fun)
print("Best vector:", result.x)
print("Best design:", best_design)
print("Last evaluation:", objective.history[-1])
