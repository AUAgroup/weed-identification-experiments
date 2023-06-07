import torch
import torchvision
from torch import (
    nn,
)  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
from torch import optim  # For all Optimization algorithms, SGD, Adam, etc.
import torch.nn.functional as F  # All functions that don't have any parameters

import tensorflow_datasets as tfds
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedKFold

import tensorflow as tf
from torch.optim.lr_scheduler import ReduceLROnPlateau
import csv
from torchvision import utils

from timm.loss import LabelSmoothingCrossEntropy
from torch.optim.lr_scheduler import CyclicLR

from utils.metrics_utils import check_accuracy, check_performances
from utils.top_tuning_utils import top_tuning
from utils.dataset_utils import get_dataset, get_dataloaders, get_raw_dataset
from utils.architecture_utils import get_architecture
from utils.feature_extraction_utils import extract_features
from utils.callbacks_utils import EarlyStopping

from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel

from config import (
    NUM_EPOCHS,
    NUM_EXPERIMENTS,
    PATIENCE,
    DATASET_NAME,
    USE_RESAMPLING,
    IM_SIZE,
)

import gc


gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices("GPU")
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


TUNING_MODE_CHANGE_EPOCH = 20
MONITOR_AFTER_EPOCH = 15
TRAINING_MODE_CHANGE_EPOCH = 120
MAX_LR = 2e-5
STEP_SIZE_UP = 10
LEARNING_RATES = [1e-4]  # 8e-6 # 1e-5
DROPOUT_RATES = [0.1]
MODEL_NAMES = [
    "trans_swin_v2_t",
    "trans_swin_t",
    "trans_swin_l",
    "cnn_efficientnet_v2",
    "trans_swin_v2_l",
    # "regnet"
]
# AUGMENTATION_MODES = ["auto_SVHN_ImageNet","auto_SVHN", , "color_jitter", , "auto_cifar10"]
AUGMENTATION_MODES = ["auto_SVHN"]
OPTIMIZERS = ["adamw"]
TRANSFER_LEARNING_MODE = [False, True]  # True]
WEIGHT_DECAYS = [1e-5]
CYCLIC_LR_MODES = ["exp_range"]  # , "none"]#, "triangular2",


with open(f"results_file-{DATASET_NAME}-paper.csv", mode="w") as results_file:
    results_writer = csv.writer(
        results_file, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
    )

    x, y = get_raw_dataset(DATASET_NAME)
    skf = StratifiedKFold(n_splits=5)
    for num_experiment, (train_index, test_index) in enumerate(skf.split(x, y)):
        for augmentation_mode in AUGMENTATION_MODES:
            train_dataset, val_dataset, test_dataset = get_dataset(
                x,
                y,
                train_index,
                test_index,
                augmentation_mode,
                # preprocess,
                use_resampling=USE_RESAMPLING,
            )
            num_classes = len(torch.unique(val_dataset[:][1]))
            print(num_classes)
            train_loader, val_loader, test_loader = get_dataloaders(
                train_dataset, val_dataset, test_dataset
            )
            for dropout_rate in DROPOUT_RATES:
                for transfer_learning_mode in TRANSFER_LEARNING_MODE:
                    for weight_decay in WEIGHT_DECAYS:
                        for optimizer_name in OPTIMIZERS:
                            for learning_rate in LEARNING_RATES:
                                for cyclic_lr_mode in CYCLIC_LR_MODES:
                                    for model_name in MODEL_NAMES:
                                        model, preprocess = get_architecture(
                                            model_name,
                                            dropout_rate,
                                            num_classes,
                                            transfer_learning_mode,
                                        )
                                        model = model.to(device)
                                        previous_epoch_loss = 10.0
                                        best_train_acc = 0
                                        best_val_acc = 0
                                        test_aux_acc = 0
                                        test_aux_accs = []
                                        best_val_accs = []
                                        best_val_acc_epochs = []
                                        test_aux_acc_epochs = []
                                        fine_tuned = False
                                        is_pessimistic = True
                                        pipeline_name = "-".join(
                                            [
                                                str(num_experiment),
                                                DATASET_NAME,
                                                str(USE_RESAMPLING),
                                                str(IM_SIZE),
                                                augmentation_mode,
                                                model_name,
                                                str(dropout_rate),
                                                str(transfer_learning_mode),
                                                str(optimizer_name),
                                                str(weight_decay),
                                                str(cyclic_lr_mode),
                                                str(MAX_LR),
                                                str(learning_rate),
                                            ]
                                        )
                                        print(f"[INFO] Training: {pipeline_name}")
                                        # Loss and optimizer
                                        criterion = nn.CrossEntropyLoss()
                                        # criterion = LabelSmoothingCrossEntropy()
                                        if optimizer_name == "adamw":
                                            optimizer = optim.AdamW(
                                                model.parameters(),
                                                lr=learning_rate,
                                                weight_decay=weight_decay,
                                            )
                                        elif optimizer_name == "adam":
                                            optimizer = optim.Adam(
                                                model.parameters(),
                                                lr=learning_rate,
                                                weight_decay=weight_decay,
                                            )

                                        early_stopping = EarlyStopping(
                                            tolerance=PATIENCE, max_delta=0.25
                                        )
                                        # Train Network
                                        best_epoch = 0
                                        best_epoch_aux = 0
                                        aux_val_acc = 0.0

                                        # (
                                        #    train_features,
                                        #    train_labels,
                                        #    test_features,
                                        #    test_labels,
                                        # ) = extract_features(
                                        #    model,
                                        #    model_name,
                                        #    train_loader,
                                        #    val_loader,
                                        #    test_loader,
                                        #    preprocess,
                                        #    device,
                                        # )
                                        # bayes = GaussianNB()
                                        # bayes.fit(train_features, train_labels)
                                        # pre_bayes_train_acc = accuracy_score(
                                        #    train_labels, bayes.predict(train_features)
                                        # )
                                        # pre_bayes_test_acc = accuracy_score(
                                        #    test_labels, bayes.predict(test_features)
                                        # )
                                        # xgb = XGBClassifier()
                                        # xgb.fit(train_features, train_labels)
                                        # pre_xgb_train_acc = accuracy_score(
                                        #    train_labels, xgb.predict(train_features)
                                        # )
                                        # pre_xgb_test_acc = accuracy_score(
                                        #    test_labels, xgb.predict(test_features)
                                        # )
                                        # lce = LCEClassifier(n_jobs=5, random_state=123)
                                        # lce.fit(train_features, train_labels)
                                        # pre_lce_train_acc = accuracy_score(
                                        #    train_labels, lce.predict(train_features)
                                        # )
                                        # pre_lce_test_acc = accuracy_score(
                                        #    test_labels, lce.predict(test_features)
                                        # )

                                        # del model, train_features, train_labels, test_features, test_labels
                                        # gc.collect()
                                        # model, preprocess = get_architecture(
                                        #    model_name,
                                        #    dropout_rate,
                                        #    num_classes,
                                        #    transfer_learning_mode,
                                        # )
                                        # model = model.to(device)
                                        for epoch in range(NUM_EPOCHS):
                                            losses = []
                                            model.train()
                                            for data, targets in tqdm(train_loader):
                                                # grid = utils.make_grid(data)
                                                # plt.imshow(grid.numpy().transpose((1, 2, 0)))
                                                # plt.show()
                                                # Get data to cuda if possible
                                                data = preprocess(data)
                                                # grid = utils.make_grid(data)
                                                # plt.imshow(grid.numpy().transpose((1, 2, 0)))
                                                # plt.show()
                                                data = data.to(device=device)
                                                # print(targets)
                                                targets = targets.to(device=device)

                                                # forward
                                                scores = model(data)
                                                loss = criterion(scores, targets)
                                                current_loss = loss.item()
                                                losses.append(current_loss)
                                                # backward
                                                optimizer.zero_grad()
                                                loss.backward()

                                                # gradient descent or adam step
                                                optimizer.step()

                                            current_epoch_loss = sum(losses) / len(
                                                losses
                                            )
                                            print(
                                                f"\n[INFO] Cost at epoch {epoch} is {current_epoch_loss:.5f}"
                                            )

                                            if (
                                                epoch == TUNING_MODE_CHANGE_EPOCH
                                                and not fine_tuned
                                            ):
                                                print(
                                                    f"[INFO] Unfreezing Layers at epoch: {epoch}"
                                                )
                                                print(
                                                    f"[INFO] Previous Loss: {previous_epoch_loss} Current Loss: {current_epoch_loss}"
                                                )
                                                unfreezeing_train_accuracy = (
                                                    check_accuracy(
                                                        train_loader,
                                                        model,
                                                        num_classes,
                                                        preprocess,
                                                        device,
                                                    )
                                                )
                                                print(
                                                    f"[INFO] (Unfreezing) Train accuracy: {unfreezeing_train_accuracy}"
                                                )
                                                unfreezeing_val_accuracy = (
                                                    check_accuracy(
                                                        val_loader,
                                                        model,
                                                        num_classes,
                                                        preprocess,
                                                        device,
                                                    )
                                                )
                                                print(
                                                    f"[INFO] (Unfreezing) Val accuracy: {unfreezeing_val_accuracy}"
                                                )
                                                fine_tuned = True
                                                if optimizer_name == "adamw":
                                                    optimizer = optim.AdamW(
                                                        model.parameters(),
                                                        lr=learning_rate * 0.1,
                                                        weight_decay=weight_decay,
                                                    )
                                                elif optimizer_name == "adam":
                                                    optimizer = optim.Adam(
                                                        model.parameters(),
                                                        lr=learning_rate * 0.1,
                                                        weight_decay=weight_decay,
                                                    )
                                                for param in model.parameters():
                                                    param.requires_grad = True

                                                if cyclic_lr_mode != "none":
                                                    scheduler = CyclicLR(
                                                        optimizer,
                                                        base_lr=learning_rate
                                                        * 0.1,  # Initial learning rate which is the lower boundary in the cycle for each parameter group
                                                        max_lr=MAX_LR,  # Upper learning rate boundaries in the cycle for each parameter group
                                                        step_size_up=STEP_SIZE_UP,  # Number of training iterations in the increasing half of a cycle
                                                        mode=cyclic_lr_mode,
                                                        cycle_momentum=False,
                                                        verbose=True,
                                                    )

                                            if epoch >= TRAINING_MODE_CHANGE_EPOCH:
                                                scheduler = ReduceLROnPlateau(
                                                    optimizer,
                                                    "max",
                                                    factor=0.8,
                                                    patience=PATIENCE // 2,
                                                    verbose=True,
                                                )

                                            if epoch >= MONITOR_AFTER_EPOCH:
                                                train_accuracy = check_accuracy(
                                                    train_loader,
                                                    model,
                                                    num_classes,
                                                    preprocess,
                                                    device,
                                                )
                                                print(
                                                    f"[INFO] Train accuracy: {train_accuracy}"
                                                )
                                                val_accuracy = check_accuracy(
                                                    val_loader,
                                                    model,
                                                    num_classes,
                                                    preprocess,
                                                    device,
                                                )
                                                print(
                                                    f"[INFO] Val accuracy: {val_accuracy}"
                                                )
                                                if train_accuracy > best_train_acc:
                                                    best_train_acc = train_accuracy
                                                if val_accuracy > best_val_acc:
                                                    best_val_accs.append(val_accuracy)
                                                    best_val_acc_epochs.append(epoch)
                                                    torch.save(
                                                        model,
                                                        "best_models/"
                                                        + pipeline_name
                                                        + ".pt",
                                                    )
                                                    best_val_acc = val_accuracy
                                                    best_epoch = epoch
                                                if (
                                                    val_accuracy > 0.975
                                                    and is_pessimistic
                                                ):
                                                    (
                                                        tmp_test_acc,
                                                        tmp_test_f1,
                                                        tmp_test_cm,
                                                    ) = check_performances(
                                                        test_loader,
                                                        model,
                                                        num_classes,
                                                        preprocess,
                                                        device,
                                                    )
                                                    print(
                                                        f"[INFO] Aux Acc Computed: {tmp_test_acc}"
                                                    )
                                                    if tmp_test_acc > test_aux_acc:
                                                        print("[INFO] Aux Acc Updated")
                                                        test_aux_acc = tmp_test_acc
                                                        test_aux_accs.append(
                                                            test_aux_acc
                                                        )
                                                        test_aux_acc_epochs.append(
                                                            epoch
                                                        )
                                                        aux_val_acc = val_accuracy
                                                        best_epoch_aux = epoch
                                                    if (
                                                        tmp_test_acc
                                                        < val_accuracy - 0.015
                                                    ):
                                                        is_pessimistic = False

                                                if (
                                                    epoch >= TRAINING_MODE_CHANGE_EPOCH
                                                ):  # ReduceLROnPlateau
                                                    scheduler.step(val_accuracy)
                                                elif (
                                                    epoch >= TUNING_MODE_CHANGE_EPOCH
                                                    and epoch
                                                    < TRAINING_MODE_CHANGE_EPOCH
                                                    and cyclic_lr_mode != "none"
                                                ):  # ExpRange
                                                    scheduler.step()
                                                if train_accuracy > 0.99:
                                                    early_stopping(
                                                        train_accuracy, val_accuracy
                                                    )
                                                if early_stopping.early_stop:
                                                    print(
                                                        f"[INFO] Early Stopping at epoch: {epoch}"
                                                    )
                                                    break

                                        (
                                            last_test_acc,
                                            last_test_f1,
                                            last_test_cm,
                                        ) = check_performances(
                                            test_loader,
                                            model,
                                            num_classes,
                                            preprocess,
                                            device,
                                        )
                                        (
                                            train_features,
                                            train_labels,
                                            test_features,
                                            test_labels,
                                        ) = extract_features(
                                            model,
                                            model_name,
                                            train_loader,
                                            val_loader,
                                            test_loader,
                                            preprocess,
                                            device,
                                        )

                                        (
                                            last_bayes_train_acc,
                                            last_bayes_test_acc,
                                            last_bayes_test_prec,
                                            last_over_xgb_train_acc,
                                            last_over_xgb_test_acc,
                                            last_over_xgb_test_prec,
                                            last_under_xgb_train_acc,
                                            last_under_xgb_test_acc,
                                            last_under_xgb_test_prec,
                                            last_over_svm_train_acc,
                                            last_over_svm_test_acc,
                                            last_over_svm_test_prec,
                                            last_over_svm_test_cm,
                                            last_under_svm_train_acc,
                                            last_under_svm_test_acc,
                                            last_under_svm_test_prec,
                                            last_under_svm_test_cm,
                                        ) = top_tuning(
                                            train_features,
                                            train_labels,
                                            test_features,
                                            test_labels,
                                        )
                                        model = torch.load(
                                            "best_models/" + pipeline_name + ".pt"
                                        )
                                        (
                                            best_test_acc,
                                            best_test_f1,
                                            best_test_cm,
                                        ) = check_performances(
                                            test_loader,
                                            model,
                                            num_classes,
                                            preprocess,
                                            device,
                                        )
                                        (
                                            train_features,
                                            train_labels,
                                            test_features,
                                            test_labels,
                                        ) = extract_features(
                                            model,
                                            model_name,
                                            train_loader,
                                            val_loader,
                                            test_loader,
                                            preprocess,
                                            device,
                                        )

                                        (
                                            best_bayes_train_acc,
                                            best_bayes_test_acc,
                                            best_bayes_test_prec,
                                            best_over_xgb_train_acc,
                                            best_over_xgb_test_acc,
                                            best_over_xgb_test_prec,
                                            best_under_xgb_train_acc,
                                            best_under_xgb_test_acc,
                                            best_under_xgb_test_prec,
                                            best_over_svm_train_acc,
                                            best_over_svm_test_acc,
                                            best_over_svm_test_prec,
                                            best_over_svm_test_cm,
                                            best_under_svm_train_acc,
                                            best_under_svm_test_acc,
                                            best_under_svm_test_prec,
                                            best_under_svm_test_cm,
                                        ) = top_tuning(
                                            train_features,
                                            train_labels,
                                            test_features,
                                            test_labels,
                                        )

                                        lsvc = LinearSVC(
                                            C=0.01, penalty="l1", dual=False
                                        ).fit(train_features, train_labels)
                                        model = SelectFromModel(
                                            lsvc,
                                            prefit=True,
                                            max_features=train_features.shape[1] // 2,
                                        )
                                        train_features = model.transform(train_features)
                                        test_features = model.transform(test_features)

                                        (
                                            best_fe_bayes_train_acc,
                                            best_fe_bayes_test_acc,
                                            best_fe_bayes_test_prec,
                                            best_fe_over_xgb_train_acc,
                                            best_fe_over_xgb_test_acc,
                                            best_fe_over_xgb_test_prec,
                                            best_fe_under_xgb_train_acc,
                                            best_fe_under_xgb_test_acc,
                                            best_fe_under_xgb_test_prec,
                                            best_fe_over_svm_train_acc,
                                            best_fe_over_svm_test_acc,
                                            best_fe_over_svm_test_prec,
                                            best_fe_over_svm_test_cm,
                                            best_fe_under_svm_train_acc,
                                            best_fe_under_svm_test_acc,
                                            best_fe_under_svm_test_prec,
                                            best_fe_under_svm_test_cm,
                                        ) = top_tuning(
                                            train_features,
                                            train_labels,
                                            test_features,
                                            test_labels,
                                        )

                                        del (
                                            train_features,
                                            train_labels,
                                            test_features,
                                            test_labels,
                                        )
                                        gc.collect()
                                        results_writer.writerow(
                                            [
                                                str(num_experiment),
                                                pipeline_name,
                                                str(early_stopping.early_stop),
                                                best_epoch,
                                                best_epoch_aux,
                                                str(is_pessimistic),
                                                round(train_accuracy, 4),
                                                round(val_accuracy, 4),
                                                round(best_train_acc, 4),
                                                round(best_val_acc, 4),
                                                str(best_val_acc_epochs),
                                                round(best_test_acc, 4),
                                                round(last_test_acc, 4),
                                                round(test_aux_acc, 4),
                                                str(test_aux_acc_epochs),
                                                str(test_aux_accs),
                                                round(aux_val_acc, 4),
                                                round(best_test_f1, 4),
                                                round(last_test_f1, 4),
                                                round(unfreezeing_train_accuracy, 4),
                                                round(unfreezeing_val_accuracy, 4),
                                                best_test_cm,
                                                # round(pre_bayes_train_acc, 4),
                                                # round(pre_bayes_test_acc, 4),
                                                # round(pre_xgb_train_acc, 4),
                                                # round(pre_xgb_test_acc, 4),
                                                # round(pre_lce_train_acc, 4),
                                                # round(pre_lce_test_acc, 4),
                                                round(last_bayes_train_acc, 4),
                                                round(last_bayes_test_acc, 4),
                                                round(last_bayes_test_prec, 4),
                                                round(last_over_xgb_train_acc, 4),
                                                round(last_over_xgb_test_acc, 4),
                                                round(last_over_xgb_test_prec, 4),
                                                round(last_under_xgb_train_acc, 4),
                                                round(last_under_xgb_test_acc, 4),
                                                round(last_under_xgb_test_prec, 4),
                                                round(last_over_svm_train_acc, 4),
                                                round(last_over_svm_test_acc, 4),
                                                round(last_over_svm_test_prec, 4),
                                                round(last_under_svm_train_acc, 4),
                                                round(last_under_svm_test_acc, 4),
                                                round(last_under_svm_test_prec, 4),
                                                round(best_bayes_train_acc, 4),
                                                round(best_bayes_test_acc, 4),
                                                round(best_bayes_test_prec, 4),
                                                round(best_over_xgb_train_acc, 4),
                                                round(best_over_xgb_test_acc, 4),
                                                round(best_over_xgb_test_prec, 4),
                                                round(best_under_xgb_train_acc, 4),
                                                round(best_under_xgb_test_acc, 4),
                                                round(best_under_xgb_test_prec, 4),
                                                round(best_over_svm_train_acc, 4),
                                                round(best_over_svm_test_acc, 4),
                                                round(best_over_svm_test_prec, 4),
                                                round(best_under_svm_train_acc, 4),
                                                round(best_under_svm_test_acc, 4),
                                                round(best_under_svm_test_prec, 4),
                                                round(best_fe_bayes_train_acc, 4),
                                                round(best_fe_bayes_test_acc, 4),
                                                round(best_fe_bayes_test_prec, 4),
                                                round(best_fe_over_xgb_train_acc, 4),
                                                round(best_fe_over_xgb_test_acc, 4),
                                                round(best_fe_over_xgb_test_prec, 4),
                                                round(best_fe_under_xgb_train_acc, 4),
                                                round(best_fe_under_xgb_test_acc, 4),
                                                round(best_fe_under_xgb_test_prec, 4),
                                                round(best_fe_over_svm_train_acc, 4),
                                                round(best_fe_over_svm_test_acc, 4),
                                                round(best_fe_over_svm_test_prec, 4),
                                                str(best_fe_over_svm_test_cm),
                                                round(best_fe_under_svm_train_acc, 4),
                                                round(best_fe_under_svm_test_acc, 4),
                                                round(best_fe_under_svm_test_prec, 4),
                                                str(best_fe_under_svm_test_cm),
                                                # round(lce_train_acc, 4),
                                                # round(lce_test_acc, 4),
                                            ]
                                        )
                                        results_file.flush()
                                        del model
                                        gc.collect()
