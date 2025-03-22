import numpy as np
from pathlib import Path
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import shutil

#### For test purposes
# path = Path(r'C:\Users\Michał\Desktop\WSB\SEM II\ZMSI\R_CNN\data\extracted_features\train\Apple')
# path = path / "fvec1013_0_x390_y434_400_489_0.npy"
#
# loaded = np.load(str(path))
# print(loaded)
#
# flattened_features = loaded.reshape(loaded.shape[0], loaded.shape[1] * loaded.shape[2] * loaded.shape[3])
# print(loaded.shape)
# print(flattened_features.shape)

# TODO It has to be tested on /test and /val directory
def train_svc_model(path_to_base_dataset : Path):
    path = path_to_base_dataset
    if not path.exists():
        raise FileNotFoundError(f"The file {path_to_base_dataset} does not exist.")

    # Make a list of files in "input" directory
    l_files = [file for file in path.iterdir() if file.is_file()]

    X = np.zeros((len(l_files), 62720))     # 62720 = 7 * 7 * 1280
    y = np.zeros((len(l_files)))

    print(X.shape)
    print(y.shape)

    base_folder = Path("../../models")
    type_of_model = "svm"
    class_name = path.name
    path_to_destination_folder = base_folder / type_of_model / class_name

    if not path_to_destination_folder.exists():
        path_to_destination_folder.mkdir(parents=True, exist_ok=True)
        print(f"Folder {path_to_destination_folder} was created.")
    else:
        print(f"Folder {path_to_destination_folder} already exist.")

    # Iteration through all files ???which are vectors??
    for i, fname in enumerate(l_files):
        if i == 1000:
            break

        loaded = np.load(str(fname))
        flattened_features = loaded.reshape(loaded.shape[0], loaded.shape[1] * loaded.shape[2] * loaded.shape[3])
        X[i] = flattened_features

        label = fname.name.rsplit(".", 1)[0]        # labels look like 'fvec1013_0_x390_y434_400_489_0'
        y[i] = label[-1:]       # Last character represents number of class

        #### For test purposes
        # print(X[i])
        # print(fname.name, y[i])

    #### For test purposes
    # print(X)
    # print(y)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X[0:1000], y[0:1000], test_size=0.2, random_state=42)

    ## I approach - simple SVC
    # model = SVC(kernel='linear')
    # model.fit(X_train, y_train)
    # y_pred = model.predict(X_test)

    ## II approach - use of grid search
    param_grid = {
        # 'C': [0.1, 1, 10, 100],
        # 'gamma': [1, 0.1, 0.01, 0.001],
        'C': [1, 10],
        'gamma': [0.1, 0.01],
        'kernel': ['rbf']
    }
    grid_search = GridSearchCV(svm.SVC(), param_grid, refit=True, verbose=2, cv=5)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}, for {class_name}")

    # The trained model is saved to pickle format for use in later stages of the project
    new_fname = "svc_for_" + class_name.lower() + "_acc_" + str(accuracy) + "_len_" + str(len(l_files)) + ".pkl"
    temp_path_to_write = path_to_destination_folder / new_fname

    # print(str(temp_path_to_write))
    with open(str(temp_path_to_write), 'wb') as f:
    # with open(new_fname, 'wb') as f:
        pickle.dump(best_model, f)

parent_path = Path(__file__).resolve().parent.parent.parent
extracted_ftrs_dir = parent_path / "data" / "extracted_features" / "train"
train_svc_model(extracted_ftrs_dir / "Apple")