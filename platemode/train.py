from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, MaxPool2D, Flatten, Dropout, Dense, Input
from keras.optimizers import Adam
from keras.backend.tensorflow_backend import set_session
from keras.utils.vis_utils import model_to_dot
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

from IPython.display import SVG
from genplate import *

def gen(batch_size=32):
    while True:
        l_plateStr,l_plateImg = G.genBatch(batch_size, 2, range(31,65),(272,72))
        X = np.array(l_plateImg, dtype=np.uint8)
        ytmp = np.array(list(map(lambda x: [M_strIdx[a] for a in list(x)], l_plateStr)), dtype=np.uint8)
        y = np.zeros([ytmp.shape[1],batch_size,len(chars)])
        for batch in range(batch_size):
            for idx,row_i in enumerate(ytmp[batch]):
                y[idx,batch,row_i] = 1

        yield X, [yy for yy in y]




adam = Adam(lr=0.001)

input_tensor = Input((72, 272, 3))
x = input_tensor
for i in range(3):
    x = Conv2D(32*2**i, (3, 3), activation='relu')(x)
    x = Conv2D(32*2**i, (3, 3), activation='relu')(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
x = Flatten()(x)
x = Dropout(0.25)(x)

n_class = len(chars)
x = [Dense(n_class, activation='softmax', name='c%d'%(i+1))(x) for i in range(7)]
model = Model(inputs=input_tensor, outputs=x)
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

# SVG(model_to_dot(model=model, show_layer_names=True, show_shapes=True).create(prog='dot', format='svg'))


best_model = ModelCheckpoint("chepai_best.h5", monitor='val_loss', verbose=0, save_best_only=True)

model.fit_generator(gen(32), steps_per_epoch=2000, epochs=5,
                    validation_data=gen(32), validation_steps=1280,
                    callbacks=[best_model])



