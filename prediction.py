import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from cnn import config
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os
import cv2
from os import listdir
from os.path import isfile, join
import warnings
warnings.filterwarnings('ignore')
tf.compat.v1.logging.set_verbosity(tf.logging.WARN)
tf.compat.v1.logging.set_verbosity(tf.logging.ERROR)

BS = 32
# trainAug = ImageDataGenerator(
#     rescale=1 / 255.0,
#     rotation_range=20,
#     zoom_range=0.05,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     shear_range=0.05,
#     horizontal_flip=True,
#     vertical_flip=True,
#     fill_mode="nearest")

# initialize the validation (and testing) data augmentation object
valAug = ImageDataGenerator(rescale=1 / 255.0)

testGen = valAug.flow_from_directory(
    config.TEST_PATH,
    class_mode="categorical",
    target_size=(50, 50),
    color_mode="rgb",
    shuffle=False,
    batch_size=BS)

model = load_model('BreastCancerModel.h5')
class_labels = testGen.class_indices
class_labels = {v: k for k, v in class_labels.items()}
classes = list(class_labels.values())

# prediction
# testing
model = load_model('BreastCancerModel.h5')


def draw_test(name, pred, im, true_label):
    WHITE = [255, 255, 255]
    expanded_image = cv2.copyMakeBorder(im, 160, 0, 0, 500, cv2.BORDER_CONSTANT, value=WHITE)
    cv2.putText(expanded_image, "Prediction- " + pred, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(expanded_image, "Actual- " + true_label, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow(name, expanded_image)


def getRandomImage(path, img_width, img_height):
    """function loads a random images from a random folder in our test path """
    folders = list(filter(lambda x: os.path.isdir(os.path.join(path, x)), os.listdir(path)))
    random_directory = np.random.randint(0, len(folders))
    path_class = folders[random_directory]
    file_path = path + path_class
    file_names = [f for f in listdir(file_path) if isfile(join(file_path, f))]
    random_file_index = np.random.randint(0, len(file_names))
    image_name = file_names[random_file_index]
    final_path = file_path + "/" + image_name
    return image.load_img(final_path, target_size=(img_width, img_height)), final_path, path_class


# dimensions of our images
img_width, img_height = 50, 50

files = []
predictions = []
true_labels = []
# predicting images
import time
from tqdm import tqdm

for i in tqdm(range(0, 5)):
    time.sleep(0.5)
    path = 'Datasets/idc/testing/'
    img, final_path, true_label = getRandomImage(path, img_width, img_height)
    files.append(final_path)
    true_labels.append(true_label)
    x = image.img_to_array(img)
    x = x * 1. / 255
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    classes = model.predict_classes(images, batch_size=10)
    predictions.append(classes)

for i in range(0, len(files)):
    image = cv2.imread((files[i]))
    draw_test("Prediction", class_labels[predictions[i][0]], image, true_labels[i])
    cv2.waitKey(0)

cv2.destroyAllWindows()
