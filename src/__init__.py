import zntrack
import typing as t
import dataclasses
from pathlib import Path


class CreateDataset(zntrack.Node):
    params: dict = zntrack.params()
    method: t.Literal["make_moons", "make_circles", "linearly_separable"] = (
        zntrack.params()
    )

    x: t.Any = zntrack.outs()
    y: t.Any = zntrack.outs()

    def run(self) -> None:
        from sklearn.datasets import make_circles, make_classification, make_moons
        import numpy as np

        if self.method == "make_moons":
            self.x, self.y = make_moons(**self.params)
        elif self.method == "make_circles":
            self.x, self.y = make_circles(**self.params)
        elif self.method == "linearly_separable":
            X, y = make_classification(**self.params)
            rng = np.random.RandomState(2)
            X += 2 * rng.uniform(size=X.shape)
            self.x, self.y = X, y
        else:
            raise ValueError(f"Unknown method: {self.method}")


class TrainTestSplit(zntrack.Node):
    x: t.Any = zntrack.deps()
    y: t.Any = zntrack.deps()

    x_train: t.Any = zntrack.outs()
    y_train: t.Any = zntrack.outs()
    x_test: t.Any = zntrack.outs()
    y_test: t.Any = zntrack.outs()

    test_size: float = zntrack.params(0.4)
    random_state: int = zntrack.params(42)

    def run(self) -> None:
        from sklearn.model_selection import train_test_split

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x, self.y, test_size=self.test_size, random_state=self.random_state
        )


@dataclasses.dataclass
class Model:
    method: t.Literal[
        "KNeighborsClassifier",
        "SVC",
        "GaussianProcessClassifier",
        "DecisionTreeClassifier",
        "RandomForestClassifier",
        "MLPClassifier",
        "AdaBoostClassifier",
        "GaussianNB",
        "QuadraticDiscriminantAnalysis",
    ]
    params: dict = dataclasses.field(default_factory=dict)
    name: str|None = None

    def __post_init__(self):
        if self.name is None:
            self.name = self.method

    def get_model(self):
        from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
        from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
        from sklearn.gaussian_process import GaussianProcessClassifier
        from sklearn.gaussian_process.kernels import RBF
        from sklearn.naive_bayes import GaussianNB
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.neural_network import MLPClassifier
        from sklearn.svm import SVC
        from sklearn.tree import DecisionTreeClassifier

        if self.method == "KNeighborsClassifier":
            return KNeighborsClassifier(**self.params)
        elif self.method == "SVC":
            return SVC(**self.params)
        elif self.method == "GaussianProcessClassifier":
            kernel = 1.0 * RBF(1.0)
            return GaussianProcessClassifier(kernel=kernel, **self.params)
        elif self.method == "DecisionTreeClassifier":
            return DecisionTreeClassifier(**self.params)
        elif self.method == "RandomForestClassifier":
            return RandomForestClassifier(**self.params)
        elif self.method == "MLPClassifier":
            return MLPClassifier(**self.params)
        elif self.method == "AdaBoostClassifier":
            return AdaBoostClassifier(**self.params)
        elif self.method == "GaussianNB":
            return GaussianNB(**self.params)
        elif self.method == "QuadraticDiscriminantAnalysis":
            return QuadraticDiscriminantAnalysis(**self.params)
        else:
            raise ValueError(f"Unknown method: {self.method}")


class Classifier(zntrack.Node):
    model: Model = zntrack.deps()

    x: t.Any = zntrack.deps()

    x_train: t.Any = zntrack.deps()
    y_train: t.Any = zntrack.deps()

    x_test: t.Any = zntrack.deps()
    y_test: t.Any = zntrack.deps()

    metrics: dict = zntrack.metrics()

    figure_path: Path = zntrack.plots_path(zntrack.nwd / "figure.png")

    def run(self):
        model = self.model.get_model()
        model.fit(self.x_train, self.y_train)

        self.metrics = {"score": model.score(self.x_test, self.y_test)}
        self.get_figure(model)

    
    def get_figure(self, clf):
        from sklearn.inspection import DecisionBoundaryDisplay
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap


        fig, ax = plt.subplots()

        x_min, x_max = self.x[:, 0].min() - 0.5, self.x[:, 0].max() + 0.5
        y_min, y_max = self.x[:, 1].min() - 0.5, self.x[:, 1].max() + 0.5

        cm = plt.cm.RdBu
        cm_bright = ListedColormap(["#FF0000", "#0000FF"])

        DecisionBoundaryDisplay.from_estimator(clf, self.x, cmap=cm, alpha=0.8, ax=ax, eps=0.5)


        # Plot the training points
        ax.scatter(
            self.x_train[:, 0], self.x_train[:, 1], c=self.y_train, cmap=cm_bright, edgecolors="k"
        )
        # Plot the testing points
        ax.scatter(
            self.x_test[:, 0],
            self.x_test[:, 1],
            c=self.y_test,
            cmap=cm_bright,
            edgecolors="k",
            alpha=0.6,
        )

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(())
        ax.set_yticks(())

        self.figure_path.parent.mkdir(exist_ok=True, parents=True)
        fig.savefig(self.figure_path)
