from pathlib import Path
import cv2
import numpy
import numpy as np
from PIL import Image
from sklearn.svm import SVC

from keras._tf_keras.keras.applications import EfficientNetB0
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.preprocessing.image import img_to_array

def extracted_feature_generator(path_to_base_dataset : Path):
    path = path_to_base_dataset
    if not path.exists():
        raise FileNotFoundError(f"The file {path_to_base_dataset} does not exist.")

    # CNN model for feature extraction
    conv_base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    feature_extractor = Model(inputs=conv_base.input, outputs=conv_base.layers[-1].output)

    print(conv_base.summary())
    print(feature_extractor.summary())
    # input()

    base_folder = path.parent.parent.parent
    type_of_dataset = path.parent.name      # "train", "test" or "val"
    class_name = path.name
    extracted_subfolder = "extracted_features"
    path_to_new_folder = base_folder / extracted_subfolder / type_of_dataset / class_name

    if not path_to_new_folder.exists():
        path_to_new_folder.mkdir(parents=True, exist_ok=True)
        print(f"Folder {path_to_new_folder} was created.")
    else:
        print(f"Folder {path_to_new_folder} already exist.")

    # Make a list of files in "input" directory
    l_files = [file for file in path.iterdir() if file.is_file()]
    # print(l_files)
    # Iteration through all files
    for i, fname in enumerate(l_files):
        if i == 10000:
            break
        print(i, fname)

        img = Image.open(str(fname))
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        #### For test purposes
        # cv2.imshow("Img", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        resized_img = numpy.resize(img, (224, 224, 3))
        input_arr = np.expand_dims(img_to_array(resized_img), axis=0)

        features = feature_extractor.predict(input_arr)     # Output features have (1,7,7,1280) shape

        #### For test purposes
        # print(path.name)
        # print(path.parent)
        # print(path.parent.name)
        # print(path.parent.parent.parent.parent)

        file_name = fname.name[3:]      # Cut out the "img" prefix
        file_name = file_name.rsplit(".", 1)[0]     # Cut out the image format like ".jpg" for example
        new_fname = "fvec" + file_name + ".npy"     # Build new name with "fvec" prefix
        temp_path_to_write = path_to_new_folder / new_fname
        # print(str(temp_path_to_write))
        np.save(str(temp_path_to_write), features)

    print(f"{len(l_files)} features data were generated")

parent_path = Path(__file__).resolve().parent.parent.parent
roi_dir = parent_path / "data" / "roi_data" / "train"
extracted_feature_generator(roi_dir / "Apple")