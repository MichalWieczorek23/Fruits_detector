from pathlib import Path
import pickle
import numpy as np

def calculate_tp_svm(path_to_test_dataset : Path, path_to_model : Path, verbose: bool = True) -> int:
    with open(path_to_model, "rb") as file:
        svc_classifier = pickle.load(file)

    positive_samples_count = 0          # How many positive samples are in the data
    tp_samples_count = 0                # How many samples could be classified as tp

    for i, f in enumerate(path_to_test_dataset.iterdir()):
        if i == 100:
            break
        if verbose:
            print("Sample i:", i)

        # Load features to X_test
        loaded = np.load(f)
        features = loaded['features']
        flattened_features = features.reshape(features.shape[0],
                                              features.shape[1] * features.shape[2] * features.shape[3])
        X_test = flattened_features

        # Get the class label from file name and store as a y_test
        label = f.name.rsplit(".", 1)[0]  # labels look like 'fvec1013_0_x390_y434_400_489_0'
        y_test = int(label[-1:])

        if y_test != 0:
            positive_samples_count += 1
            # Predict y_pred based on X_test
            y_pred = svc_classifier.predict(flattened_features)[0]
            if y_pred != 0:
                tp_samples_count += 1
    if verbose:
        print("Positive: ", positive_samples_count)
        print("Tp: ", tp_samples_count)
    return tp_samples_count

def calculate_tn_svm(path_to_test_dataset : Path, path_to_model : Path, verbose: bool = True) -> int:
    with open(path_to_model, "rb") as file:
        svc_classifier = pickle.load(file)

    negative_samples_count = 0          # How many negative samples are in the data
    tn_samples_count = 0                # How many samples could be classified as tn

    for i, f in enumerate(path_to_test_dataset.iterdir()):
        if i == 100:
            break
        if verbose:
            print("Sample i:", i)

        # Load features to X_test
        loaded = np.load(f)
        features = loaded['features']
        flattened_features = features.reshape(features.shape[0],
                                              features.shape[1] * features.shape[2] * features.shape[3])
        X_test = flattened_features

        # Get the class label from file name and store as a y_test
        label = f.name.rsplit(".", 1)[0]  # labels look like 'fvec1013_0_x390_y434_400_489_0'
        y_test = int(label[-1:])

        if y_test == 0:
            negative_samples_count += 1
            # Predict y_pred based on X_test
            y_pred = svc_classifier.predict(flattened_features)[0]
            if y_pred == 0:
                tn_samples_count += 1

    if verbose:
        print("Negative: ", negative_samples_count)
        print("Tn: ", tn_samples_count)
    return tn_samples_count


def calculate_fn_svm(path_to_test_dataset : Path, path_to_model : Path, verbose: bool = True) -> int:
    with open(path_to_model, "rb") as file:
        svc_classifier = pickle.load(file)

    positive_samples_count = 0          # How many positive samples are in the data
    fn_samples_count = 0                # How many samples could be classified as fn

    for i, f in enumerate(path_to_test_dataset.iterdir()):
        if i == 100:
            break
        if verbose:
            print("Sample i:", i)

        # Load features to X_test
        loaded = np.load(f)
        features = loaded['features']
        flattened_features = features.reshape(features.shape[0],
                                              features.shape[1] * features.shape[2] * features.shape[3])
        X_test = flattened_features

        # Get the class label from file name and store as a y_test
        label = f.name.rsplit(".", 1)[0]  # labels look like 'fvec1013_0_x390_y434_400_489_0'
        y_test = int(label[-1:])

        if y_test != 0:
            positive_samples_count += 1
            # Predict y_pred based on X_test
            y_pred = svc_classifier.predict(flattened_features)[0]
            if y_pred == 0:
                fn_samples_count += 1
    if verbose:
        print("Positive: ", positive_samples_count)
        print("Fn: ", fn_samples_count)
    return fn_samples_count


def calculate_fp_svm(path_to_test_dataset : Path, path_to_model : Path, verbose: bool = True) -> int:
    with open(path_to_model, "rb") as file:
        svc_classifier = pickle.load(file)

    negative_samples_count = 0          # How many negative samples are in the data
    fp_samples_count = 0                # How many samples could be classified as fp

    for i, f in enumerate(path_to_test_dataset.iterdir()):
        if i == 100:
            break
        if verbose:
            print("Sample i:", i)

        # Load features to X_test
        loaded = np.load(f)
        features = loaded['features']
        flattened_features = features.reshape(features.shape[0],
                                              features.shape[1] * features.shape[2] * features.shape[3])
        X_test = flattened_features

        # Get the class label from file name and store as a y_test
        label = f.name.rsplit(".", 1)[0]  # labels look like 'fvec1013_0_x390_y434_400_489_0'
        y_test = int(label[-1:])

        if y_test == 0:
            negative_samples_count += 1
            # Predict y_pred based on X_test
            y_pred = svc_classifier.predict(flattened_features)[0]
            if y_pred != 0:
                fp_samples_count += 1

    if verbose:
        print("Negative: ", negative_samples_count)
        print("Fp: ", fp_samples_count)
    return fp_samples_count

def calculate_acc_svm(path_to_test_dataset: Path, path_to_model: Path, verbose: bool = True) -> float:
    tp = calculate_tp_svm(path_to_test_dataset, path_to_model)
    tn = calculate_tn_svm(path_to_test_dataset, path_to_model)
    fp = calculate_fp_svm(path_to_test_dataset, path_to_model)
    fn = calculate_fn_svm(path_to_test_dataset, path_to_model)
    acc = (tp + tn) / (tp + tn + fp + fn)
    if verbose:
        print("acc:", acc)
    return acc

# def calculate_precission():
#
# def calculate_recall():
#
# def calculate_f1_score():

parent_path = Path(__file__).resolve().parent.parent.parent
test_data = parent_path / "data" / "extracted_features" / "train"
model_path = parent_path / "models" / "svm"

calculate_acc_svm(test_data / "Apple", model_path / "Apple" / "svc_for_apple_acc_0.9793333333333333_len_15000.pkl")