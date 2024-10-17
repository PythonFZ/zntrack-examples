import zntrack.examples

project = zntrack.Project()

with project:
    zntrack.examples.ParamsToMetrics(
        params={"loss": 0.1, "accuracy": 0.9},
    )

project.repro()