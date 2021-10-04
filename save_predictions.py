import time
import tensorflow as tf
import numpy as np
from tensorflow.keras import optimizers
from sklearn.metrics import accuracy_score

from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)

x_train, x_test = x_train / 255-0.5, x_test / 255-0.5




y_train=tf.keras.utils.to_categorical(y_train,10)
y_test=tf.keras.utils.to_categorical(y_test,10)
x_train=np.expand_dims(x_train,axis=-1)
x_test=np.expand_dims(x_test,axis=-1)

#ntrain = y_test.shape[0]
nnets = 3
#P = np.zeros(shape=(ntrain,nnets))
supermodel=[]
for i in range(nnets):
        start = time.perf_counter()
        
        model = tf.keras.models.load_model("model"+str(i))
        supermodel.append(model)
        print(i,'acc:',accuracy_score(np.argmax(y_test,axis=1),np.argmax(model.predict(x_test),axis=1)))
        
        time_elapsed = time.perf_counter() - start
        print(f"Time: {time_elapsed}")
        
P=np.asarray([a.predict(x_test) for a in supermodel])

final_value = accuracy_score(np.argmax(y_test,axis=1),np.argmax(np.mean(P,axis=0),axis=1)) # 20 models stack accurasy                     
print(final_value)
#import pdb; pdb.set_trace()

np.savez('Pmat.npz', Pmatr=P)                  
