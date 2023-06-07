from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from lce import LCEClassifier
from sklearn.svm import SVC


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.model_selection import StratifiedKFold


def compute_performance(
    model, train_features, train_labels, test_features, test_labels
):

    train_acc = accuracy_score(train_labels, model.predict(train_features))
    test_acc = accuracy_score(test_labels, model.predict(test_features))
    test_prec = precision_score(
        test_labels, model.predict(test_features), average="macro"
    )
    test_cm = confusion_matrix(
        test_labels, model.predict(test_features), normalize="true"
    )

    return train_acc, test_acc, test_prec, test_cm


def top_tuning(train_features, train_labels, test_features, test_labels):
    skf = StratifiedKFold(n_splits=3)
    over_xgb_parameters = {
        "max_depth": [5],
        "n_estimators": [500],
        "learning_rate": [0.1, 0.01],
        "subsample": [0.75],
    }

    under_xgb_parameters = {
        "max_depth": [2],
        "n_estimators": [100],
        "learning_rate": [0.1, 0.01],
        "subsample": [0.5],
    }

    over_svm_parameters = {"C": [1, 10], "kernel": ["rbf"]}
    under_svm_parameters = {"C": [0.1, 1], "kernel": ["rbf"]}

    bayes = GaussianNB()
    bayes.fit(train_features, train_labels)
    bayes_train_acc, bayes_test_acc, bayes_test_prec, test_cm = compute_performance(
        bayes, train_features, train_labels, test_features, test_labels
    )

    xgb = XGBClassifier()
    over_grid_search = GridSearchCV(
        estimator=xgb,
        param_grid=over_xgb_parameters,
        scoring="balanced_accuracy",
        n_jobs=10,
        cv=skf,
        verbose=3,
    )
    over_grid_search.fit(train_features, train_labels)
    print(over_grid_search.best_params_)
    (
        over_xgb_train_acc,
        over_xgb_test_acc,
        over_xgb_test_prec,
        test_cm,
    ) = compute_performance(
        over_grid_search.best_estimator_,
        train_features,
        train_labels,
        test_features,
        test_labels,
    )

    xgb = XGBClassifier()
    under_grid_search = GridSearchCV(
        estimator=xgb,
        param_grid=under_xgb_parameters,
        scoring="balanced_accuracy",
        n_jobs=10,
        cv=skf,
        verbose=3,
    )
    under_grid_search.fit(train_features, train_labels)
    print(under_grid_search.best_params_)
    (
        under_xgb_train_acc,
        under_xgb_test_acc,
        under_xgb_test_prec,
        test_cm,
    ) = compute_performance(
        under_grid_search.best_estimator_,
        train_features,
        train_labels,
        test_features,
        test_labels,
    )

    svm = SVC()
    over_grid_search = GridSearchCV(
        estimator=svm,
        param_grid=over_svm_parameters,
        scoring="balanced_accuracy",
        n_jobs=10,
        cv=skf,
        verbose=3,
    )
    over_grid_search.fit(train_features, train_labels)
    print(over_grid_search.best_params_)
    (
        over_svm_train_acc,
        over_svm_test_acc,
        over_svm_test_prec,
        over_svm_test_cm,
    ) = compute_performance(
        over_grid_search.best_estimator_,
        train_features,
        train_labels,
        test_features,
        test_labels,
    )

    svm = SVC()
    under_grid_search = GridSearchCV(
        estimator=svm,
        param_grid=under_svm_parameters,
        scoring="balanced_accuracy",
        n_jobs=10,
        cv=skf,
        verbose=3,
    )
    under_grid_search.fit(train_features, train_labels)
    print(under_grid_search.best_params_)
    (
        under_svm_train_acc,
        under_svm_test_acc,
        under_svm_test_prec,
        under_svm_test_cm,
    ) = compute_performance(
        under_grid_search.best_estimator_,
        train_features,
        train_labels,
        test_features,
        test_labels,
    )

    # lce = LCEClassifier(n_jobs=10, random_state=2023)
    # lce.fit(train_features, train_labels)
    # lce_train_acc = accuracy_score(
    #    train_labels, lce.predict(train_features)
    # )
    # lce_test_acc = accuracy_score(
    #    test_labels, lce.predict(test_features)
    # )

    return (
        bayes_train_acc,
        bayes_test_acc,
        bayes_test_prec,
        over_xgb_train_acc,
        over_xgb_test_acc,
        over_xgb_test_prec,
        under_xgb_train_acc,
        under_xgb_test_acc,
        under_xgb_test_prec,
        over_svm_train_acc,
        over_svm_test_acc,
        over_svm_test_prec,
        over_svm_test_cm,
        under_svm_train_acc,
        under_svm_test_acc,
        under_svm_test_prec,
        under_svm_test_cm,
    )
