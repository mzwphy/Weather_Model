#data used here is from the Solar Dynamic Observatory (SDO). There are about 949807 magnetograms of the solar active regions for the period 2010 to 2018.

import pandas as pd
import numpy as np
import os
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import glob
import cv2


df_train = pd.read_csv(
    "Train_Data_by_AR_png_224.csv",
    dtype="str"
)

# apply a cut on class1 
df_1 = df_train[df_train['class'] == "1"] 

#shuffle the active regions
df_1 = df_1.sample(frac=1).reset_index(drop=True)

# Number of rows to drop
n = 148249
# Dropping last n rows
df_1.drop(df_1.tail(n).index,inplace = True)


# apply a cut on class0 
df_0 = df_train[df_train['class'] == "0"] 

#shuffle the active regions
df_0 = df_0.sample(frac=1).reset_index(drop=True)


# Number of rows to drop
k = 609108
# Dropping last k rows
df_0.drop(df_0.tail(k).index,inplace = True)

#combine(merge) dataframes using the append method
training_df = df_1.append(df_0)
training_df = training_df.sample(frac=1).reset_index(drop=True)
print(training_df.head(20))
print(len(training_df))


df_val = pd.read_csv(
    "Validation_Data_by_AR_png_224.csv",
    dtype="str"
)

# apply a cut on class1 
df_3 = df_val[df_val['class'] == "1"] 

#shuffle the active regions
df_3 = df_3.sample(frac=1).reset_index(drop=True)

# Number of rows to drop
m = 19291
# Dropping last m rows
df_3.drop(df_3.tail(m).index,inplace = True)

# apply a cut on class1 
df_4 = df_val[df_val['class'] == "0"] 

#shuffle the active regions
df_4 = df_4.sample(frac=1).reset_index(drop=True)

# Number of rows to drop
g = 75642
# Dropping last g rows
df_4.drop(df_4.tail(g).index,inplace = True)

#combine(merge) dataframes using the append method
validation_df = df_3.append(df_4)
validation_df = validation_df.sample(frac=1).reset_index(drop=True)
print(validation_df.head(20))
print(len(validation_df))


pd.set_option('colheader_justify', 'center')


train_datagen = ImageDataGenerator(
    rescale=1./255
)


val_datagen = ImageDataGenerator(
    rescale=1/255
)

train_generator = train_datagen.flow_from_dataframe(
    training_df,
    x_col='filename',
    y_col='class',
    directory="/home/mzwandile/Desktop/Weather Model/Data/", # Put your own absolute path to image folders
    target_size=(224, 224),  # Adjust to match the size of your input images
    batch_size=32,
    class_mode='binary',  # Change to 'categorical' if you have more than two classes
    shuffle=True,
    color_mode="grayscale",
    interpolation='nearest'

)

val_generator = val_datagen.flow_from_dataframe(
    validation_df,
    x_col='filename',
    y_col='class',
    directory="/home/mzwandile/Desktop/Weather Model/Data/", 
    target_size=(224,224),
    class_mode='binary',  # Change to 'categorical' if you have more than two classes 
    batch_size=32,
    shuffle=True,
    interpolation='nearest',
    color_mode="grayscale"
)


kernel_size = (3, 3)
pool_size = (2, 2)

model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(32, kernel_size = kernel_size, activation="relu", input_shape=(224,224,1)),
    tf.keras.layers.MaxPooling2D(pool_size = pool_size),
    #tf.keras.layers.Dropout(0.5),


    tf.keras.layers.Conv2D(64, kernel_size = kernel_size, activation="relu", padding = 'same'),
    tf.keras.layers.MaxPooling2D(pool_size = pool_size),
    tf.keras.layers.Dropout(0.5),


    tf.keras.layers.Conv2D(64, kernel_size = kernel_size, activation="relu", padding = 'same'),
    tf.keras.layers.MaxPooling2D(pool_size = pool_size),
    tf.keras.layers.Dropout(0.5),


    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(250, activation="relu"),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(1, activation = 'sigmoid')

  ]
)


#Compiling
metriks = [
    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    tf.keras.metrics.TruePositives(name="TP"),
    tf.keras.metrics.TrueNegatives(name="TN"),
    tf.keras.metrics.FalsePositives(name="FP"),
    tf.keras.metrics.FalseNegatives(name="FN"),
    tf.keras.metrics.Precision(name="precision"),
    tf.keras.metrics.Recall(name="recall"),
    tf.keras.metrics.AUC(name='auc'),
    tf.keras.metrics.AUC(name='prc', curve='PR') # precision-recall curve
]

model.compile(
    loss = 'binary_crossentropy',
    optimizer = tf.keras.optimizers.RMSprop(lr = 1e-3),
    metrics = metriks

)

model.summary()




history = model.fit_generator(
    train_generator,
    epochs=100,
    verbose=1,
    validation_data = val_generator
)


colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
def plot_metrics(history):
    metrics = ['loss', 'prc', 'precision', 'recall']
    for n, metric in enumerate(metrics):
        name = metric.replace("_"," ").capitalize()
        plt.subplot(2,2,n+1)
        plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
        plt.plot(history.epoch, history.history['val_'+metric],
             color=colors[0], linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
            plt.ylim([0.8,1])
        else:
            plt.ylim([0,1])

            plt.legend()
    plt.show()
plot_metrics(history)

#Save model for testing
model.save('FlareModel.h5')



acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
los = history.history['loss']
val_los = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(los, label='Training Loss')
plt.plot(val_los, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()






