import pandas as pd
import numpy as np
import os
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn import preprocessing

df_trainingzX = pd.read_csv("dataframes/df_X_train.csv", dtype="str")
df_trainingzM = pd.read_csv("dataframes/df_M_train.csv", dtype="str")
df_trainingzC = pd.read_csv("dataframes/df_C_train.csv", dtype="str")
df_trainingz0 = pd.read_csv("dataframes/df_0_train.csv", dtype="str")

train_data_df = df_trainingzX.append([df_trainingzM, df_trainingzC, df_trainingz0])
train_data_df = train_data_df.sample(frac=1).reset_index(drop=True)

df_validationX = pd.read_csv("dataframes/df_X_val.csv", dtype="str")
df_validationM = pd.read_csv("dataframes/df_M_val.csv", dtype="str")
df_validationC = pd.read_csv("dataframes/df_C_val.csv", dtype="str")
df_validation0 = pd.read_csv("dataframes/df_0_val.csv", dtype="str")

val_data_df = df_validationX.append([df_validationM, df_validationC, df_validation0])
val_data_df = val_data_df.sample(frac=1).reset_index(drop=True)


df_testingX = pd.read_csv("dataframes/df_X_test.csv", dtype="str")
df_testingM = pd.read_csv("dataframes/df_M_test.csv", dtype="str")
df_testingC = pd.read_csv("dataframes/df_C_test.csv", dtype="str")
df_testing0 = pd.read_csv("dataframes/df_0_test.csv", dtype="str")

test_data_df = df_testingX.append([df_testingM, df_testingC, df_testing0])
test_data_df = test_data_df.sample(frac=1).reset_index(drop=True)


'''
train_data_df = pd.get_dummies(train_data_df, columns = ['Class'])
print(train_data_df.head(50))
'''



train_datagen = ImageDataGenerator(
    rescale=1./255
)


val_datagen = ImageDataGenerator(
    rescale=1/255
)


test_datagen = ImageDataGenerator(
    rescale=1/255
)





train_generator = train_datagen.flow_from_dataframe(
    train_data_df,
    x_col='filename',
    y_col= 'Class',
    directory="/home/mzwandile/Desktop/Weather Model/Images/", # Put your own absolute path to image folders
    target_size=(224, 224),  # Adjust to match the size of your input images
    batch_size=32,
    weight_col=None,
    class_mode='categorical',  # Change to 'categorical' if you have more than two classes
    shuffle=True,
    color_mode="grayscale",
    interpolation='nearest'

)

val_generator = val_datagen.flow_from_dataframe(
    val_data_df,
    x_col='filename',
    y_col='Class',
    directory="/home/mzwandile/Desktop/Weather Model/Images/", 
    target_size=(224,224),
    class_mode='categorical',  # Change to 'categorical' if you have more than two classes 
    batch_size=32,
    shuffle=True,
    interpolation='nearest',
    color_mode="grayscale"
)


test_generator = test_datagen.flow_from_dataframe(
    test_data_df,
    x_col='filename',
    y_col='Class',
    directory="/home/mzwandile/Desktop/Weather Model/Images/", 
    target_size=(224,224),
    class_mode='categorical',  # Change to 'categorical' if you have more than two classes 
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

    tf.keras.layers.Dense(4, activation = 'softmax')

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

#Compile model
model.compile(
	optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=metriks
)

model.summary()

#train the model
history = model.fit_generator(
    train_generator,
    epochs=50,
    verbose=1,
    validation_data = val_generator
)

#Save model for testing
#model.save('transFlareModel.h5')

#model.evaluate(test_generator)

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
    plt.savefig('transFlareModel_metrics.png')

plot_metrics(history)

#Visualizations
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(4, 4))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()
plt.savefig('transFlareModel_loss.png')

