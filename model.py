from keras.layers import *
from keras.models import Model
from keras import regularizers
import pickle
from keras.callbacks import *
from keras.utils import to_categorical
import numpy as np
class EventPrediction(object):
	def __init__(self,window=3,filters=300,bch=25,epoch=50,train_test_split=0.8):
		self.window=window
		self.filters=filters
		self.batch=bch
		self.epc=epoch
		self.mc=ModelCheckpoint("./checkpoints"+"weights.{epoch:02d}-loss{loss:.2f}-acc{acc:.2f}-.hdf5", monitor='loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
		#self.es=EarlyStopping(monitor='loss', min_delta=1, patience=15, verbose=1, mode='auto')
		self.tb=TensorBoard(log_dir='./logs',  batch_size=self.batch, write_graph=True, write_grads=True, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
		self.data=None
		self.split=train_test_split
		self.end=None
		self.model=None
	def load_data(self):
		choice=input("Enter number to load dataset for-\n1.large\n2.mid\n3.small")
		if(choice=='1'):
			name='large'
		elif(choice=='2'):
			name='mid'
		elif(choice=='3'):
			name='small'
		else:
			raise(Exception("Invalid input"))
		with open("./"+name+"-lms_vec_training.pkl","rb")as f:
			self.data=pickle.load(f)
		self.end=int(len(self.data['Y'])*self.split)
	def load_model(self):
		longt=Input(shape=(None,300),name='long')
		medt=Input(shape=(None,300),name='medium')
		st=Input(shape=(None,300),name="short")
		std=Dropout(0.3)(st)
		lt1=Conv1D(self.filters, self.window, input_shape=(None,300),strides=1, padding='valid', dilation_rate=1, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(longt)
		lt2=Conv1D(self.filters, 2,input_shape=(None,300), strides=1, padding='valid', dilation_rate=1, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(lt1)
		lt2d=Dropout(0.3)(lt2)
		mt1=Conv1D(self.filters, self.window, input_shape=(None,300),strides=1, padding='valid', dilation_rate=1, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(medt)
		mt2=Conv1D(self.filters, 2,strides=1,input_shape=(None,300), padding='valid', dilation_rate=1, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(mt1)
		mt2d=Dropout(0.3)(mt2)
		ltoutput=GlobalMaxPooling1D()(lt2d)
		mtoutput=GlobalMaxPooling1D()(mt2d)
		stoutput=GlobalAveragePooling1D()(st)
		feature=concatenate([ltoutput,mtoutput,stoutput])
		h1=Dense(100,activation='relu')(feature)
		h2=Dense(50,activation='relu')(h1)
		trend=Dense(2,activation='sigmoid',name='class')(h2)
		self.model=Model(inputs=[longt,medt,st],outputs=[trend])
		self.model.compile(
			optimizer='adam',
			loss='categorical_crossentropy',
			metrics=['accuracy'])
		self.model.summary()

	def train(self):
		
		self.model.fit({"long":self.data['X']['long'][:self.end],"medium":self.data['X']['medium'][:self.end],"short":self.data['X']['short'][:self.end]},
		{'class':to_categorical(self.data['Y'][:self.end],num_classes=2)},epochs=self.epc,batch_size=self.batch,callbacks=[self.mc,self.tb])
	def test(self,data=None):
		print("#"*5+"  Testing  "+"#"*5)
		if(data==None):
			data=self.data
		loss_metrics=self.model.evaluate({"long":data['X']['long'][self.end:],"medium":data['X']['medium'][self.end:],"short":data['X']['short'][self.end:]},
		{'class':to_categorical(data['Y'][self.end:],num_classes=2)},batch_size=1)
		for i,j in enumerate(loss_metrics):
			print(self.model.metrics_names[i],j)
predict=EventPrediction()
predict.load_data()
predict.load_model()
predict.train()
predict.test()