import os
import gdown
import tensorflow as tf
from deepface.basemodels import VGGFace
from deepface.commons import functions

# -------------------------------------
# pylint: disable=line-too-long
# -------------------------------------
# dependency configurations

tf_version = int(tf.__version__.split(".", maxsplit=1)[0])

if tf_version == 1:
    from keras.models import Model, Sequential
    from keras.layers import Convolution2D, Flatten, Activation
elif tf_version == 2:
    from tensorflow.keras.models import Model, Sequential
    from tensorflow.keras.layers import Convolution2D, Flatten, Activation
# -------------------------------------

# Labels for the genders that can be detected by the model.
labels = ["Woman", "Man"]

def loadModel(
    url="https://github.com/serengil/deepface_models/releases/download/v1.0/gender_model_weights.h5",
    model_path = None
):

    model = VGGFace.baseModel()

    # --------------------------

    classes = 2
    base_model_output = Sequential()
    base_model_output = Convolution2D(classes, (1, 1), name="predictions")(model.layers[-4].output)
    base_model_output = Flatten()(base_model_output)
    base_model_output = Activation("softmax")(base_model_output)

    # --------------------------

    gender_model = Model(inputs=model.input, outputs=base_model_output)
	
    return gender_model
################################################################################################
    # --------------------------

    # load weights

    home = functions.get_deepface_home()
    if model_path == None:
        model_path = home + "/.deepface/weights/gender_model_weights.h5"
        
    if os.path.isfile(home + "/.deepface/weights/gender_model_weights.h5") != True:
        print("gender_model_weights.h5 will be downloaded...")

        output = home + "/.deepface/weights/gender_model_weights.h5"
        gdown.download(url, output, quiet=False)
    print(home + "/.deepface/weights/gender_model_weights.h5")
    print("Model will be loaded from : "+str(model_path))
    gender_model.load_weights(model_path)

    return gender_model
