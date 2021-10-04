# Ramunas Girdziusas, May 7th, 2021

import warnings  
with warnings.catch_warnings():  
    warnings.filterwarnings("ignore",category=FutureWarning)
    import tensorflow as tf

#print(tf.__version__)

    import numpy as np
    from tensorflow.keras import optimizers
    from sklearn.metrics import accuracy_score
    from tensorflow.keras.datasets import mnist

import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)

x_train, x_test = x_train / 255-0.5, x_test / 255-0.5

y_train=tf.keras.utils.to_categorical(y_train,10)
y_test=tf.keras.utils.to_categorical(y_test,10)
x_train=np.expand_dims(x_train,axis=-1)
x_test=np.expand_dims(x_test,axis=-1)

#ntrain = y_test.shape[0]
nnets = 3
data = np.load('Pmatrix.npz')
P = data['Pmatr']
labels_ideal = np.argmax(y_test,axis=1)
labels_pred = np.argmax(np.mean(P,axis=0),axis=1)
#labels_pred = np.argmax(np.max(P,axis=0),axis=1) #same thing

final_value = accuracy_score(labels_ideal, labels_pred)                     
print(final_value)

#The best second prediction
ns = y_test.shape[0]
Pav = np.mean(P,axis=0)
inds_max = np.argmax(Pav,axis=1)
Pav[range(ns),inds_max] = np.NINF
labels_secondary = np.argmax(Pav,axis=1)

#Error indices
inds = np.nonzero(labels_ideal-labels_pred)

#For debugging:
#A = np.random.rand(5, 5)
#import pdb; pdb.set_trace()

fig, axs = plt.subplots(2, 9, figsize=(12, 3))

for i in range(2):
    for j in range(9):
        it = i*9+j 
        ind = inds[0][it] #[0] as an array is wrapped into a tuple
        axs[i,j].imshow(x_test[ind,:,:,0], interpolation='bicubic', 
                        cmap=plt.cm.get_cmap('Greys'))
        axs[i,j].get_xaxis().set_visible(False)
        axs[i,j].get_yaxis().set_visible(False)
        if it<0:
            axs[i,j].set_title("T="+str(labels_ideal[ind])+" "
                                   +"P1="+str(labels_pred[ind])+" "
                                   +"P2="+str(labels_secondary[ind]))
        else:
            axs[i,j].set_title(str(labels_ideal[ind]) 
                               + " " + str(labels_pred[ind]) 
                               + " " + str(labels_secondary[ind]))
        axs[i,j].grid(False)

plt.show()
