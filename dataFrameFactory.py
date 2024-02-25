import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
import os

dir_1 = os.getcwd()
data_dir = dir_1 + "/Images/"
print("")
print("data path: = ", data_dir)
print("")


df = pd.read_csv(
	"C1.0_24hr_224_png_Labels.csv",
	dtype = "str",
	names=["filename", "klass"]
)


Class = []
for i in range(len(df)):
    if df["klass"][i].startswith("M") == True:
        Class.append('M')
    elif df["klass"][i].startswith("C") == True:
        Class.append('C')
    elif df["klass"][i].startswith("X") == True:
        Class.append('X')
    else:
        Class.append('0')

df_new = pd.DataFrame({
        "filename": df["filename"],
        "Class": Class
    }
)
pd.set_option('colheader_justify', 'center')


# apply a cut on class to get X events 
df_X = df_new[df_new['Class'] == "X"]
df_X = df_X.sample(frac=1).reset_index(drop=True) 
print(df_X.head(50))
print(len(df_X))

# apply a cut on class to get M events 
df_M = df_new[df_new['Class'] == "M"] 

#shuffle the active regions
df_M = df_M.sample(frac=1).reset_index(drop=True)

# Number of rows to drop
n = 23680
# Dropping last n rows
df_M.drop(df_M.tail(n).index,inplace = True)
print(df_M.head(50))
print(len(df_M))


# apply a cut on class to get M events 
df_C = df_new[df_new['Class'] == "C"]

#shuffle the active regions
df_C = df_C.sample(frac=1).reset_index(drop=True)

# Number of rows to drop
q = 158059
# Dropping last q rows
df_C.drop(df_C.tail(q).index,inplace = True)
print(df_C.head(50))
print(len(df_C))

# apply a cut on class to get "0" events 
df_0 = df_new[df_new['Class'] == "0"]

#shuffle the active regions
df_0 = df_0.sample(frac=1).reset_index(drop=True)

# Number of rows to drop
v = 756465
# Dropping last v rows
df_0.drop(df_0.tail(v).index,inplace = True)
print(df_0.head(50))
print(len(df_0))


####################################################################################
####################################################################################
"""
#select events
df_X_train = df_X.iloc[0:1500]
df_X_val = df_X.iloc[1500:2500]
df_X_test = df_X.iloc[2500:2543]

#save to csv
df_X_train.to_csv("df_X_train.csv", index = False)
df_X_val.to_csv("df_X_val.csv", index = False)
df_X_test.to_csv("df_X_test.csv", index = False)



#select events
df_M_train = df_M.iloc[0:1500]
df_M_val = df_M.iloc[1500:2500]
df_M_test = df_M.iloc[2500:3000]

#save to csv
df_M_train.to_csv("df_M_train.csv", index = False)
df_M_val.to_csv("df_M_val.csv", index = False)
df_M_test.to_csv("df_M_test.csv", index = False)



#select events
df_C_train = df_C.iloc[0:1500]
df_C_val = df_C.iloc[1500:2500]
df_C_test = df_C.iloc[2500:3000]

#save to csv
df_C_train.to_csv("df_C_train.csv", index = False)
df_C_val.to_csv("df_C_val.csv", index = False)
df_C_test.to_csv("df_C_test.csv", index = False)



#select events
df_0_train = df_0.iloc[0:1500]
df_0_val = df_0.iloc[1500:2500]
df_0_test = df_0.iloc[2500:3000]

#save to csv
df_0_train.to_csv("df_0_train.csv", index = False)
df_0_val.to_csv("df_0_val.csv", index = False)
df_0_test.to_csv("df_0_test.csv", index = False)
"""
###################################################################################
###################################################################################


