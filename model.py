from keras.layers import *
from keras.models import Model
from keras import regularizers
import pickle
from keras.callbacks import *
window=3
filters=300
import numpy as np
batch=25
epc=80
mc=ModelCheckpoint("./checkpoints"+"weights.{epoch:02d}-loss{loss:.2f}-acc{acc:.2f}-.hdf5", monitor='loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
es=EarlyStopping(monitor='loss', min_delta=1, patience=15, verbose=1, mode='auto')
tb=TensorBoard(log_dir='./logs',  batch_size=batch, write_graph=True, write_grads=True, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
longt=Input(shape=(None,300),name='long')
medt=Input(shape=(None,300),name='medium')
st=Input(shape=(None,300),name="short")
lt1=Conv1D(filters, window, input_shape=(None,300),strides=1, padding='valid', dilation_rate=1, activation='relu', use_bias=True, kernel_initializer='glorot_normal', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(longt)
lt2=Conv1D(300, 2,input_shape=(None,300), strides=1, padding='valid', dilation_rate=1, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(lt1)
mt1=Conv1D(filters, window, input_shape=(None,300),strides=1, padding='valid', dilation_rate=1, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(medt)
mt2=Conv1D(300, 2,strides=1,input_shape=(None,300), padding='valid', dilation_rate=1, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(mt1)
ltoutput=GlobalMaxPooling1D()(lt2)
mtoutput=GlobalMaxPooling1D()(mt2)
stoutput=GlobalAveragePooling1D()(st)
feature=concatenate([ltoutput,mtoutput,stoutput])
h1=Dense(100,activation='relu')(feature)
h2=Dense(50,activation='relu')(h1)
trend=Dense(2,activation='sigmoid',name='class')(h2)
model=Model(inputs=[longt,medt,st],outputs=[trend])
model.compile(
	optimizer='adam',
	loss='categorical_crossentropy',
	metrics=['accuracy'])
model.summary()
with open("./mid-lms_vec_training.pkl","rb")as f:
	dat=pickle.load(f)
model.fit({"long":np.asarray(dat['X']['long']),"medium":dat['X']['medium'],"short":dat['X']['short']},
	{'class':dat['Y']},epochs=epc,batch_size=batch,callbacks=[mc,tb,es])