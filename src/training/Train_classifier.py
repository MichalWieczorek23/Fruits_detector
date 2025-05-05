import numpy as np
from pathlib import Path
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import shutil
from keras._tf_keras.keras.applications import EfficientNetB0
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
import keras
import matplotlib.pyplot as plt

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

number_of_files = 15000

# TODO It has to be tested on /test and /val directory
def train_svc_model(path_to_base_dataset : Path):
    path = path_to_base_dataset
    if not path.exists():
        raise FileNotFoundError(f"The file {path_to_base_dataset} does not exist.")

    # Make a list of files in "input" directory
    l_files = [file for file in path.iterdir() if file.is_file()]

    # X = np.zeros((len(l_files), 62720), dtype=np.float16)     # 62720 = 7 * 7 * 1280
    # y = np.zeros((len(l_files)))
    X = np.zeros((number_of_files, 62720), dtype=np.float16)  # 62720 = 7 * 7 * 1280
    y = np.zeros((number_of_files), dtype=np.uint8)

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
        if i == number_of_files:
            break

        loaded = np.load(str(fname))
        features = loaded['features']
        flattened_features = features.reshape(features.shape[0], features.shape[1] * features.shape[2] * features.shape[3])
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
    # X_train, X_test, y_train, y_test = train_test_split(X[0:number_of_files], y[0:number_of_files], test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Data has been loaded...")

    ## I approach - simple SVC
    model = svm.SVC(kernel='linear', C=0.01)  # From gridsearch we know that best kernel is 'linear' and best C is 0.01
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # ## II approach - use of grid search
    # param_grid = {
    #     'C': [0.01, 0.1, 1],
    #     'kernel': ['linear']
    # }
    # grid_search = GridSearchCV(
    #     svm.SVC(),
    #     param_grid=param_grid,
    #     refit=True,
    #     # n_jobs=-1,
    #     verbose=2,
    #     cv=2)
    # grid_search.fit(X_train, y_train)
    #
    # best_model = grid_search.best_estimator_
    # best_score = grid_search.best_score_
    # best_params = grid_search.best_params_
    # print("Best_model: ", best_model)
    # print("Best_score: ", best_score)
    # print("Best_params: ", best_params)
    # y_pred = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}, for {class_name}")

    # The trained model is saved to pickle format for use in later stages of the project
    new_fname = "svc_for_" + class_name.lower() + "_acc_" + str(accuracy) + "_len_" + str(number_of_files) + ".pkl"
    temp_path_to_write = path_to_destination_folder / new_fname

    # print(str(temp_path_to_write))
    with open(str(temp_path_to_write), 'wb') as f:
    # with open(new_fname, 'wb') as f:
        pickle.dump(model, f)


def separate_files_to_bckgrnd_class_f(path_to_base_dataset : Path):
    path = path_to_base_dataset
    if not path.exists():
        raise FileNotFoundError(f"The file {path_to_base_dataset} does not exist.")

    temp_classes_dict = {
        "Apple": 1,
        "Banana": 2,
        "Carrot": 3,
        "Orange": 4
    }

    object_dir = path / str(temp_classes_dict[path.name])
    bckgrnd_dir = path / str(0)
    object_dir.mkdir(exist_ok=True)
    bckgrnd_dir.mkdir(exist_ok=True)

    for f in path.glob("*.jpg"):
        fname = f.name
        name_parts = fname.split("_")
        if len(name_parts) > 6:
            try:
                class_str = name_parts[-1].split(".")[0]
                cls = int(class_str)
                target_dir = path / str(cls)
                shutil.move(str(f), str(target_dir))
                print(f"{fname} was moved to {cls}/ directory")
            except ValueError:
                print(f"Unable to identify class for file: {fname}")
            except IndexError:
                print(f"Incorrect file name format: {fname}")
        else:
            print(f"Invalid file format: {fname}")

def train_mlp_model(
        path_to_train_dataset : Path,
        path_to_val_dataset : Path,
        path_to_cnn_models : Path,
        display_plot : bool = True,
        save_model : bool = True):
    if not path_to_train_dataset.exists():
        raise FileNotFoundError(f"The file {path_to_train_dataset} does not exist.")
    if not path_to_val_dataset.exists():
        raise FileNotFoundError(f"The file {path_to_val_dataset} does not exist.")

    datagen = ImageDataGenerator(
        rescale=1. / 255,
    )

    train_dataset = datagen.flow_from_directory(
        path_to_train_dataset,
        target_size=(224, 224),
        batch_size=20,
        class_mode='binary',
        shuffle=True
    )

    val_dataset = datagen.flow_from_directory(
        path_to_val_dataset,
        target_size=(224, 224),
        batch_size=20,
        class_mode='binary',
        shuffle=True
    )

    # # Calculate the number of steps for only 20% of the data:
    # train_steps = int(0.2 * len(train_dataset))
    # val_steps = int(0.2 * len(val_dataset))

    # Load the base model
    conv_base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    conv_base.trainable = False

    # Model architecture of a new part
    # inputs = keras.Input(shape=(224, 224, 3))
    # x = conv_base(inputs, training=False)
    # x = keras.layers.GlobalAveragePooling2D()(x)
    # x = keras.layers.Dropout(0.3)(x)
    # x = keras.layers.Dense(128, activation='relu')(x)
    # x = keras.layers.Dropout(0.2)(x)
    # outputs = keras.layers.Dense(1)(x)
    # model = keras.Model(inputs, outputs)

    inputs = keras.Input(shape=(224, 224, 3))
    x = conv_base(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Dense(1)(x)
    model = keras.Model(inputs, outputs)

    model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=1e-5),
                  loss=keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=[keras.metrics.BinaryAccuracy()])

    # Model training
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        steps_per_epoch=100,
        validation_steps=50,
        epochs=100
    )
    # history = model.fit(
    #     train_dataset,
    #     validation_data=val_dataset,
    #     epochs=10
    # )


    if save_model:
        epochs = len(history.history['loss'])
        final_val_acc = history.history['val_binary_accuracy'][-1]
        model_filename = f'mlp_acc_{final_val_acc}_epochs_{str(epochs)}.h5'
        model.save(model_filename)
        path_to_cnn_models.mkdir(exist_ok=True)
        shutil.move(model_filename, path_to_cnn_models)

    if display_plot:
        plt.figure(figsize=(12, 5))

        # Loss chart
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Train loss')
        plt.plot(history.history['val_loss'], label='Val loss')
        plt.title("Loss during training")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        # Chart accuracy
        plt.subplot(1, 2, 2)
        plt.plot(history.history['binary_accuracy'], label='Train acc')
        plt.plot(history.history['val_binary_accuracy'], label='Val acc')
        plt.title("Accuracy during training")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()

        plt.tight_layout()
        plt.show()

parent_path = Path(__file__).resolve().parent.parent.parent
extracted_ftrs_dir = parent_path / "data" / "extracted_features" / "train"
roi_data_train_dir = parent_path / "data" / "roi_data" / "train"
roi_data_val_dir = parent_path / "data" / "roi_data" / "val"

# train_svc_model(extracted_ftrs_dir / "Orange")
# separate_files_to_bckgrnd_class_f(roi_data_val_dir / "Apple")
roi_data_train_dir = parent_path / "data" / "roi_data" / "train" / "Apple"
roi_data_val_dir = parent_path / "data" / "roi_data" / "val" / "Apple"
models_cnn_dir = parent_path / "models" / "cnn" / "Apple"
train_mlp_model(roi_data_train_dir, roi_data_val_dir, models_cnn_dir)