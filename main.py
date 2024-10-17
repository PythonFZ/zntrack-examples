import zntrack.examples

project = zntrack.Project()

with project:
    zntrack.examples.ParamsToOuts(
        params="Hello, World!",
    )

project.repro()