import zntrack.examples
import numpy as np

project = zntrack.Project()

with project:
    zntrack.examples.ParamsToMetrics(
        params={"loss": 0.1, "accuracy": 0.9},
    )
with project.group("plots"):
    zntrack.examples.WritePlots(
        x=list(range(10)),
        y=list(range(10)),
    )
    zntrack.examples.WritePlots(
        x=list(range(10)),
        y=np.exp(np.linspace(0, 1, 10)).tolist(),
    )


project.repro()