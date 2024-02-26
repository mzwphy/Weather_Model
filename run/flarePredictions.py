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
from tensorflow.keras.preprocessing import image
from tqdm import tqdm
from tensorflow.keras.utils import plot_model




df_testingX = pd.read_csv("../dataframes/df_X_test.csv", dtype="str")
df_testingM = pd.read_csv("../dataframes/df_M_test.csv", dtype="str")
df_testingC = pd.read_csv("../dataframes/df_C_test.csv", dtype="str")
df_testing0 = pd.read_csv("../dataframes/df_0_test.csv", dtype="str")

test_data_df = df_testingX.append([df_testingM, df_testingC, df_testing0])
test_data_df = test_data_df.sample(frac=1).reset_index(drop=True)



test_datagen = ImageDataGenerator(
    rescale=1/255
)



test_generator = test_datagen.flow_from_dataframe(
    test_data_df,
    x_col='filename',
    y_col='Class',
    directory="/home/mzwandile/Desktop/Weather Model/Images/", 
    target_size=(224,224),
    class_mode='categorical',  # Change to 'categorical' if you have more than two classes 
    batch_size=1,
    shuffle=True,
    interpolation='nearest',
    color_mode="grayscale"
)

#convert turple (test_generator) to list
test_data = []

for b in range(1543):
    item = test_generator[b]
    test_data.append(item)

print(len(test_data))

x_test = []
y_test = []

for x, y in test_data:
    x_test.append(x)
    y_test.append(y)

print(y_test[0])

#load trained model
model = tf.keras.models.load_model(os.getcwd()+"/transFlareModel.h5")
plot_model(model, to_file="modelSketch.png", show_shapes=True)
y_pred = model.predict(x_test)
y_pred.flatten()
print("Predicted classes : ", y_pred)

x_axis = ["0", "C", "M", "X"]
plt.figure(figsize=(5, 4))
plt.bar(x_axis, y_pred.flatten())
plt.plot(x_axis, y_pred.flatten(), marker="*", markersize=18, linestyle="", color="r")
plt.xlabel("Solar Flare Category")
plt.ylabel("Probability of Occurance")
plt.title("Solar Flare Prediction")
plt.ylim([-0.1,1.0])
plt.savefig('prediction.jpg')
plt.show()


