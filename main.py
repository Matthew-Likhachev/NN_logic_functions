from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
import matplotlib.pyplot as plt
import numpy as np
coloumn = 4
rows = coloumn**2

input_data = [[0 for i in range(coloumn) ] for j in range(rows)]
output_data = [[0] for i in range(rows) ]
#print((output_data))
for i in range(rows):
    if ((i//8)%2==0):
        input_data[i][0]=0

    else:
        input_data[i][0] = 1

    if((i//4)%2==0):
        input_data[i][1] = 0
    else:
        input_data[i][1] = 1
    if ((i // 2) % 2 == 0):
        input_data[i][2] = 0
    else:
        input_data[i][2] = 1
    if (i % 2 == 0):
        input_data[i][3] = 0
        output_data[i] = input_data[i][3]
    else:
        input_data[i][3] = 1
        output_data[i] = input_data[i][3]

check_x = [0 for i in range(2) ]
check_x[0]=input_data.pop(0)
check_x[1]=input_data.pop(14)
check_y = [[0] for i in range(2) ]
check_y[0]=output_data.pop(0)
check_y[1]=output_data.pop(14)
#print(check_y)
#convert list to numpy array
input_data = np.array(input_data)
output_data = np.array(output_data)
#create a model
model = Sequential()
model.add(Dense(4, activation="tanh"))
#model.add(Dense(4, activation="tanh"))
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer="sgd", loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(input_data,output_data,validation_split=0.5, batch_size=1, epochs=80)
print(model.predict(check_x))
# [[0.00891833]
#  [0.9930017 ]]

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()



# #initial data  already prepared
# input_x=np.array([[0,0],[0,1],[1,0],[1,1]])
# output_y =np.array( [[0],[1],[1],[0]])
#
# #use model sequential for categorical network
# model = Sequential()
# #add some layers
# #    0 - \
# #         - 0 -
# #    0 - /
# model.add(Dense(2   ,input_dim=2))
# model.add(Activation('tanh'))
# model.add(Dense(1))
# model.add(Activation('sigmoid'))
#
# model.compile(optimizer ='sgd' , loss='binary_crossentropy')
# model.fit(input_x,output_y,  batch_size=1, epochs=2000)
