# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd 
import numpy as np 
import matplotlib.pylab as plt
import seaborn as sns


# %%
import tensorflow as tf
print(tf.__version__)


# %%
columns_names=['sepal_length','sepal_width','petal_length','petal_width','class']
raw_dataset=pd.read_csv('data/iris.data.csv',names=columns_names)
raw_dataset.tail()


# %%
sns.pairplot(raw_dataset,hue='class',palette="muted",size=5,vars=['sepal_width','sepal_length','petal_width','petal_length'],kind='scatter')


# %%
sns.violinplot(x='class',y='sepal_length',data=raw_dataset)


# %%
sns.violinplot(x='class',y='sepal_width',data=raw_dataset)


# %%
sns.violinplot(x='class',y='petal_length',data=raw_dataset)


# %%
sns.violinplot(x='class',y='petal_width',data=raw_dataset)


# %%
dataset=pd.get_dummies(raw_dataset,prefix='',prefix_sep='')
dataset.tail()


# %%
train_dataset=dataset.sample(frac=.8,random_state=0)
test_dataset=dataset.drop(train_dataset.index)


# %%
train_features=train_dataset[['sepal_length','sepal_width','petal_length','petal_width']]
train_labels=train_dataset[['Iris-setosa','Iris-versicolor','Iris-virginica']]

test_features=test_dataset[['sepal_length','sepal_width','petal_length','petal_width']]
test_label=test_dataset[['Iris-setosa','Iris-versicolor','Iris-virginica']]


# %%
train_features.shape


# %%
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers.experimental import preprocessing


# %%
normalizer=preprocessing.Normalization(input_shape=[4,])


# %%
normalizer.adapt(np.asarray(train_features))


# %%
normalizer.mean.numpy()


# %%
print('original',np.asarray(train_features[:1]))
print()
print('normalized',normalizer(np.asarray(train_features[:1])).numpy())


# %%
model=Sequential()
model.add(normalizer)
for i in range(7):
    model.add(Dense(8,activation='relu'))
model.add(Dense(3,activation='softmax'))


# %%
model.summary()


# %%
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=.001),loss='categorical_crossentropy', metrics=['accuracy'])


# %%
model.predict(train_features[:10])


# %%
history=model.fit(
    train_features,
    train_labels,
    validation_split=.2,
    epochs=200
)


# %%
from plot_helper import plot_confusion_matrix, plt_accuracy_learning_curve, plt_loss_learning_curve


# %%
plt_accuracy_learning_curve(history,full_limit=True)
plt_loss_learning_curve(history)


# %%
predict_classes=model.predict_classes(test_features)
predict_classes=pd.Series(predict_classes)
predict_classes=predict_classes.map({0:'Iris-setosa',1:'Iris-versicolor',2:'Iris-virginica'})
predict_classes


# %%
true_classes=test_label.idxmax(axis=1)
true_classes


# %%
from sklearn.metrics import accuracy_score
accuracy_score(true_classes,predict_classes)


# %%
plot_confusion_matrix(true_classes,predict_classes,['Iris-setosa','Iris-versicolor','Iris-virginica'])


# %%



