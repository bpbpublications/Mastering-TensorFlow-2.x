#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pathlib
import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds


# In[19]:


data_dir = '/Users/rdua/Downloads/alpine'
data_dir = pathlib.Path(data_dir)


# In[20]:




batch_size = 33
img_height = 3024
img_width = 4032

def get_alpine_dataset(data_dir, batch_size, img_height, img_width):
    data_dir = pathlib.Path(data_dir)
    image_count = len(list(data_dir.glob('*/*.jpg')))
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
          data_dir,
          validation_split=0.2,
          subset="training",
          seed=123,
          image_size=(img_height, img_width),
          batch_size=batch_size)
    return train_ds


def show_data(ds):
    image_batch, label_batch = next(iter(ds))

    plt.figure(figsize=(10, 10))
    for i in range(9):
      ax = plt.subplot(3, 3, i + 1)
      plt.imshow(image_batch[i].numpy().astype("uint8"))
      #label = label_batch[i]
      #plt.title(class_names[label])
      plt.axis("off")
#show_data(train_ds_1)


# In[ ]:




