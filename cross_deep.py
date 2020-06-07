import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras.backend as K

def df_to_dataset(dataframe,shuffle=True,batch_size=10,predict_column='gender'):
    dataframe = dataframe.copy()
    labels = dataframe.pop(predict_column)
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe),labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size,drop_remainder=True)
    del dataframe
    return ds

def preprocess(data,select_columns,embed_dim,batch_size,is_train=True,predict_column='gender'):
    data = data.replace(r'\N',0).replace(np.nan,0)
    data[select_columns] = data[select_columns].astype('int32')
    feature_columns = build_column(data,select_columns,embed_dim,predict_column)
    if is_train:
        train_data,val_data = train_test_split(data)
        train_ds = df_to_dataset(train_data,batch_size=batch_size,predict_column=predict_column)
        val_ds = df_to_dataset(val_data,batch_size=batch_size,predict_column=predict_column)
        return train_ds,val_ds,feature_columns
    else:
        test_ds = tf.data.Dataset.from_tensor_slices((dict(data)))
        test_ds = test_ds.batch(batch_size,drop_remainder=False)
        return test_ds,feature_columns

def build_column(data,select_columns,embed_dim,predict_column):
    num_buckets = {}
    columns = []
    l = locals()
    for i in select_columns:
        if i == predict_column or i=='click_times':
            continue
        num_buckets[i] = data[i].value_counts().shape[0]
        locals()
#         if i == 'ad_id':
#             l[i] = tf.feature_column.categorical_column_with_hash_bucket(i,hash_bucket_size=1000000,dtype=tf.dtypes.int32)
#         else:
        l[i] = tf.feature_column.categorical_column_with_identity(i,num_buckets=num_buckets[i],default_value=0)
        columns.append(tf.feature_column.embedding_column(l[i],embed_dim))
    click_times = tf.feature_column.numeric_column('click_times',shape=1,dtype=tf.dtypes.float32)
    columns.append(click_times)
    return columns
	
class CrossDeep(tf.keras.Model):
    def __init__(self,cross_layer_num,deep_layer_num,deep_layer_dim,batch_size,feature_columns,embed_dim,num_classes=10,**kwargs):
        super(CrossDeep,self).__init__(name='CrossDeep')
        self.cross_layer_num = cross_layer_num
        self.dense_feature = layers.DenseFeatures(feature_columns)
        self.deep_layer_num = deep_layer_num
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.deep_layer_dim = deep_layer_dim
        self.embed_dim = embed_dim
        self.inputs_shape = 5*self.embed_dim+1
        self.W = []
        self.bias = []
        self.dense_list = []
#         self.dense = layers.Dense(self.num_classes,activation='sigmoid')
        self.softmax = layers.Softmax(input_shape=(449,))
        for i in range(self.deep_layer_num):
            self.dense_list.append(layers.Dense(self.deep_layer_dim,activation='relu'))
        b_init = tf.zeros_initializer()
        w_init = tf.random_normal_initializer()
        for i in range(self.cross_layer_num):
            self.W.append(tf.Variable(initial_value=w_init(shape=(self.batch_size,self.inputs_shape)),trainable=True))
            self.bias.append(tf.Variable(initial_value=b_init(shape=(self.batch_size,1)),trainable=True))
#             self.W.append(self.add_weight(name=f'cross_w_{i}',shape=[None,self.inputs_shape]),trainable=True,initializer='glorot_uniform')
#             self.bias.append(self.add_weight(name=f'cross_b_{i}',shape=[None,self.inputs_shape]),trainable=True,initializer='zero')
    def call(self,inputs,training=True):
        inputs = self.dense_feature(inputs)
        for i in range(self.cross_layer_num):
            if i==0:
                cross = inputs
            cross = layers.Add()([K.dot(K.dot(inputs,tf.transpose(cross)),self.W[i]),self.bias[i],cross])
        for i in range(self.deep_layer_num):
            if i==0:
                deep = inputs
            deep = self.dense_list[i](deep)
        deep = K.reshape(deep,shape=(self.batch_size,self.deep_layer_dim))
        result = self.softmax(layers.Concatenate()([cross,deep]))
#         result = K.reshape(K.argmax(result,1),shape=(self.batch_size,1))
        return result
		
def find_top_k(data,topk = 1):
    user_topk = data.sort_values('click_times',ascending=False).drop_duplicates('user_id',keep='first').drop('Unnamed: 0',axis=1)
    return user_topk
	
def train(data,select_columns,embed_dim,cross_layer_num,deep_layer_num,deep_layer_dim,batch_size,learning_rate,epochs,predict_column):
    train_ds,val_ds,feature_columns = preprocess(data,select_columns,embed_dim,batch_size,is_train=True,predict_column='gender')
#     feature_columns = build_column(data,select_columns,embed_dim,predict_column)
    model = CrossDeep(cross_layer_num,deep_layer_num,deep_layer_dim,batch_size,feature_columns,embed_dim)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    model.fit(train_ds,epochs=epochs,validation_data=val_ds)
    return model

def test(data,model,select_columns,embed_dim,cross_layer_num,deep_layer_num,deep_layer_dim,batch_size,learning_rate,epochs,predict_column):
    select_columns.remove(predict_column)
    test_ds,feature_columns = preprocess(data,select_columns,embed_dim,batch_size,is_train=False,predict_column='gender')
#     feature_columns = build_column(data,select_columns,embed_dim,predict_column)
    result = model.predict(test_ds)
    return result
	
if __name__ == '__main__':
	data = pd.read_csv(your_file)
	select_columns = []
	model = train(data,select_columns,embed_dim=12,cross_layer_num=8,deep_layer_num=8,deep_layer_dim=128,batch_size=32,learning_rate=5e-4,epochs=3,predict_column='gender')