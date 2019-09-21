# USAGE
# python train_model.py

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.optimizers import Adagrad
from keras.utils import np_utils
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from cnn.cancernet import CancerNet
from cnn import config
from cnn import header
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# =============================================================================
#  construct the argument parser and parse the arguments
# =============================================================================
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--plot", type=str, default="plot.png",
                help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

# initialize our number of epochs, initial learning rate, and batch
# size
NUM_EPOCHS = 1
INIT_LR = 1e-2
BS = 32

# determine the total number of image paths in training, validation,
# and testing directories
trainPaths = list(paths.list_images(config.TRAIN_PATH))
totalTrain = len(trainPaths)
totalVal = len(list(paths.list_images(config.VAL_PATH)))
totalTest = len(list(paths.list_images(config.TEST_PATH)))

# account for skew in the labeled data
trainLabels = [int(p.split(os.path.sep)[-2]) for p in trainPaths]
trainLabels = np_utils.to_categorical(trainLabels)
classTotals = trainLabels.sum(axis=0)
classWeight = classTotals.max() / classTotals

# initialize the training training data augmentation object
trainAug = ImageDataGenerator(
    rescale=1 / 255.0,
    rotation_range=20,
    zoom_range=0.05,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.05,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="nearest")

# initialize the validation (and testing) data augmentation object
valAug = ImageDataGenerator(rescale=1 / 255.0)

# initialize the training generator
trainGen = trainAug.flow_from_directory(
    config.TRAIN_PATH,
    class_mode="categorical",
    target_size=(50, 50),
    color_mode="rgb",
    shuffle=True,
    batch_size=BS)

# initialize the validation generator
valGen = valAug.flow_from_directory(
    config.VAL_PATH,
    class_mode="categorical",
    target_size=(50, 50),
    color_mode="rgb",
    shuffle=False,
    batch_size=BS)

# initialize the testing generator
testGen = valAug.flow_from_directory(
    config.TEST_PATH,
    class_mode="categorical",
    target_size=(50, 50),
    color_mode="rgb",
    shuffle=False,
    batch_size=BS)

# initialize our CancerNet model and compile it
model = CancerNet.build(width=50, height=50, depth=3,classes=2)
opt = Adagrad(lr=INIT_LR, decay=INIT_LR / NUM_EPOCHS)
#linear decay learning rate is used.lr_rate decrease on each epoch
model.compile(loss="binary_crossentropy", optimizer=opt,metrics=["accuracy"])

model.summary()
   # fit the model
H = model.fit_generator(
        trainGen,
        steps_per_epoch=totalTrain // BS,
        validation_data=valGen,
        validation_steps=totalVal // BS,
        class_weight=classWeight,
        epochs=NUM_EPOCHS)
model.save("BreastCancerModel.h5")

batch_size = 32
img_row, img_height, img_depth = 50, 50, 3

class_labels = valGen.class_indices
class_labels = {v: k for k, v in class_labels.items()}
classes = list(class_labels.values())
    
nb_train_samples = totalTrain
nb_validation_samples = totalVal
    
 
Y_pred = model.predict_generator(valGen, nb_validation_samples // batch_size + 1)
y_pred = np.argmax(Y_pred, axis=1)
    
target_names = list(class_labels.values())
    
plt.figure(figsize=(20, 20))
cnf_matrix = confusion_matrix(valGen.classes, y_pred)
    
plt.imshow(cnf_matrix, interpolation='nearest')
plt.colorbar()
tick_marks = np.arange(len(classes))
_ = plt.xticks(tick_marks, classes, rotation=90)
_ = plt.yticks(tick_marks, classes)
print("[INFO] evaluating network...")
testGen.reset()
predIdxs = model.predict_generator(testGen,steps=(totalTest // BS) + 1)
   
# # for each image in the testing set we need to find the index of the
   # # label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)
   # # show a nicely formatted classification report
print(classification_report(testGen.classes, predIdxs,target_names=testGen.class_indices.keys()))
   # # compute the confusion matrix and and use it to derive the raw
   # # accuracy, sensitivity, and specificity
cm = confusion_matrix(testGen.classes, predIdxs)
total = sum(sum(cm))
acc = (cm[0, 0] + cm[1, 1]) / total
sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    
    # show the confusion matrix, accuracy, sensitivity, and specificity
print(cm)
print("acc: {:.4f}".format(acc))
print("sensitivity: {:.4f}".format(sensitivity))
print("specificity: {:.4f}".format(specificity))

 # plot the training loss and accuracy
# TYPE 1 GRAPH
N = NUM_EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
    
plt.savefig('graph.png')
plt.savefig(args["plot"])
plt.savefig('type1Graph.png')

#type2 graph
import matplotlib.pyplot as plt


acc = H.history['acc']
val_acc = H.history['val_acc']

loss = H.history['loss']
val_loss = H.history['val_loss']

epochs_range = range(N)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig('type2Graph.png')
plt.show()

    # visualization of confusion matrix and classification report
    # =============================================================================
import matplotlib.pyplot as plt
cm =  [[0.50, 1.00],
       [0.00, 0.00]]
# =============================================================================
# PUT YOUR OWN CONFUSION MATRIX
# =============================================================================

labels = ['class 0', 'class 1', 'class 2']
fig, ax = plt.subplots()
h = ax.matshow(cm)
fig.colorbar(h)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual truth')
    
    
    
import matplotlib.pyplot as plt
import numpy as np
import itertools
    
    
def plot_classification_report(classificationReport,title='Classification report',cmap='RdBu'):
    
        classificationReport = classificationReport.replace('\n\n', '\n')
        classificationReport = classificationReport.replace(' / ', '/')
        lines = classificationReport.split('\n')
    
        classes, plotMat, support, class_names = [], [], [], []
        for line in lines[1:]:  # if you don't want avg/total result, then change [1:] into [1:-1]
            t = line.strip().split()
            if len(t) < 2:
                continue
            classes.append(t[0])
            v = [float(x) for x in t[1: len(t) - 1]]
            support.append(int(t[-1]))
            class_names.append(t[0])
            plotMat.append(v)
    
        plotMat = np.array(plotMat)
        xticklabels = ['Precision', 'Recall', 'F1-score']
        yticklabels = ['{0} ({1})'.format(class_names[idx], sup)
                       for idx, sup in enumerate(support)]
    
        plt.imshow(plotMat, interpolation='nearest', cmap=cmap, aspect='auto')
        plt.title(title)
        plt.colorbar()
        plt.xticks(np.arange(3), xticklabels, rotation=45)
        plt.yticks(np.arange(len(classes)), yticklabels)
    
        upper_thresh = plotMat.min() + (plotMat.max() - plotMat.min()) / 10 * 8
        lower_thresh = plotMat.min() + (plotMat.max() - plotMat.min()) / 10 * 2
        for i, j in itertools.product(range(plotMat.shape[0]), range(plotMat.shape[1])):
            plt.text(j, i, format(plotMat[i, j], '.2f'),
                     horizontalalignment="center",
                     color="white" if (plotMat[i, j] > upper_thresh or plotMat[i, j] < lower_thresh) else "black")
    
        plt.ylabel('Metrics')
        plt.xlabel('Classes')
        plt.tight_layout()
    
    
def main():
        sampleClassificationReport = """             precision    recall  f1-score   support
    
              0:Benign       0.93      0.84      0.88        79814
              1:Malignant       0.68      0.83      0.75        31196
              macro_avg       0.80      0.84      0.81       111010
              weighted_avg       0.86      0.84      0.85       111010
              """
        plot_classification_report(sampleClassificationReport)
        plt.show()
        plt.close()
        plt.savefig('newclassreport.jpg')
    
    
if __name__ == '__main__':
        main()
    