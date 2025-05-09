import numpy as np

import zntrack.examples

project = zntrack.Project()

with project:
    zntrack.examples.ParamsToMetrics(
        params={"loss": 0.1, "accuracy": 0.9},
    )

with project.group("outs"):
    zntrack.examples.ParamsToOuts(params="Lorem ipsum", name="StringOuts")
    zntrack.examples.ParamsToOuts(params=42, name="NumericOuts")
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
