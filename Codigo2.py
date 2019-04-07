
import numpy as np
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
%matplotlib inline

train_path = 'dataset/training_set'
valid_path = 'dataset/valid'
test_path = 'dataset/test_set'

train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size=(128,128), classes = ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16'], batch_size=10)
valid_batches = ImageDataGenerator().flow_from_directory(valid_path, target_size=(128,128), classes = ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16'], batch_size=4)
test_batches = ImageDataGenerator().flow_from_directory(test_path, target_size=(128,128), classes = ['0','1','2','3','4','5','6'.'7','8','9','10','11','12','13','14','15','16'], batch_size=10)

#Plot images with albes within jupyter notebook
def plots(ims, figsize=(12,6), rows=1, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none')

imgs, labels = next(train_batches)

plots(imgs,titles=labels)

model = Sequential([
    Conv2D(32, (3,3), activation ='relu', input_shape=(128,128,3)),
    Flatten(),
    Dense(2, activation='softmax'),
])

model.add(Convolution2D(32, 3, 3, activation ='relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.compile(Adam(lr=0.0001), loss ='categorical_crossentropy', metrics =['accuracy'])

model.fit_generator(train_batches, steps_per_epoch=1000,
                   validation_data = valid_batches, validation_steps=250, epochs =10, verbose=2)

test_imgs, test_labels = next(test_batches)
plots(test_imgs, titles =test_labels)

test_labels = test_labels[:,0]

test_labels

predictions = model.predict_generator(test_batches, steps=1, verbose=0)

predictions

cm = confusion_matrix(test_labels, predictions[:,0])

def plot_confusion_matrix(cm, classes,
                         normalize = False,
                         title = 'Confusion matrix',
                         cmap =plt.cm.Blues):
    """
    Esta función imprime y grafica la matriz de confusion
    Normalizacion puede ser aplicada poniendo 'normalize = True'
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")
        
    print(cm)
    
    thresh = cm.max()/2
    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j,i,cm[i,j],
                horizontalalignment='center',
                color="white" if cm[i,j]>thresh else "black")
        
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

cm_plot_labels = ['cat','dog']
plot_confusion_matrix(cm, cm_plot_labels, title='Confusion Matrix')
