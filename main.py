import zntrack

zntrack.config.ALWAYS_CACHE = True

from src import CreateDataset, TrainTestSplit, Model, Classifier

project = zntrack.Project()

models = [
    Model(
        method="KNeighborsClassifier",
        params={"n_neighbors": 3},
    ),
    Model(
        method="SVC",
        params={"kernel": "linear", "C": 0.025, "random_state": 42},
        name="Linear_SVM",
    ),
    Model(
        method="SVC",
        params={"gamma": 2, "C": 1, "random_state": 42},
        name="RBF_SVM",
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

with project.group("moons"):
    ds = CreateDataset(method="make_moons", params={"noise": 0.3, "random_state": 0})
    split = TrainTestSplit(x=ds.x, y=ds.y)

    for model in models:
        c = Classifier(
            model=model,
            x=ds.x,
            x_train=split.x_train,
            y_train=split.y_train,
            x_test=split.x_test,
            y_test=split.y_test,
            name=model.name,
        )

project.repro()
