import zntrack

zntrack.config.ALWAYS_CACHE = True

from src import Classifier, CombineFigures, CreateDataset, Model, TrainTestSplit

project = zntrack.Project()

models = [
    Model(
        method="KNeighborsClassifier",
        params={"n_neighbors": 3},
    ),
    Model(
        method="SVC",
        params={"kernel": "linear", "C": 0.025, "random_state": 42},
        name="Linear-SVM",
    ),
    Model(
        method="SVC",
        params={"gamma": 2, "C": 1, "random_state": 42},
        name="RBF-SVM",
    ),
    Model(
        method="GaussianProcessClassifier",
        params={"random_state": 42},
    ),
    Model(
        method="DecisionTreeClassifier",
        params={"max_depth": 5, "random_state": 42},
    ),
    Model(
        method="RandomForestClassifier",
        params={
            "max_depth": 5,
            "n_estimators": 10,
            "max_features": 1,
            "random_state": 42,
        },
    ),
    Model(
        method="MLPClassifier",
        params={"alpha": 1, "max_iter": 1000, "random_state": 42},
    ),
    Model(
        method="AdaBoostClassifier",
        params={"random_state": 42},
    ),
    Model(
        method="GaussianNB",
    ),
    Model(
        method="QuadraticDiscriminantAnalysis",
    ),
]


def classify(ds):
    split = TrainTestSplit(x=ds.x, y=ds.y)

    cfs = []

    for model in models:
        cfs.append(
            Classifier(
                model=model,
                x=ds.x,
                x_train=split.x_train,
                y_train=split.y_train,
                x_test=split.x_test,
                y_test=split.y_test,
                name=model.name,
            )
        )

    CombineFigures(cfs=cfs)


with project.group("moons"):
    ds = CreateDataset(method="make_moons", params={"noise": 0.3, "random_state": 0})
    classify(ds)

with project.group("circles"):
    ds = CreateDataset(
        method="make_circles", params={"noise": 0.2, "factor": 0.5, "random_state": 0}
    )
    classify(ds)

with project.group("linearly-separable"):
    ds = CreateDataset(
        method="linearly_separable",
        params={
            "n_features": 2,
            "n_redundant": 0,
            "n_informative": 2,
            "random_state": 42,
            "n_clusters_per_class": 1,
        },
    )
    classify(ds)

project.repro()
