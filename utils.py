import matplotlib.pyplot as plt
from random import randint
import numpy as np

def show_image(X, Y, k):
    fig = plt.figure(figsize=(10,10))
    plt.subplots(2, 4, figsize=(10,10), sharex='row',gridspec_kw={'hspace': 0, 'wspace': 0})

    idx = randint(0, X.shape[0] - 1)

    for i in list(np.linspace(1,11,k,dtype = int)):
      plt.subplot(2,k,i)
      plt.imshow(X[idx + i,:,:,:])
      plt.axis("off")
      plt.title('Image')

      plt.subplot(2,k,i+1)
      plt.imshow(Y[idx + i,:,:,0])
      plt.axis("off")
      plt.title('Mask')

    plt.subplots_adjust(wspace=0)
    plt.tight_layout()
    plt.show()


# Train Validation Split
def train_val_split(X, Y, valid_split=0.8):
    num_images = X.shape[0]
    x_train = X[:int(valid_split * num_images),:,:,:]
    y_train = Y[:int(valid_split * num_images),:,:,0]
    y_train = np.expand_dims(y_train, axis=-1)

    x_valid = X[int(valid_split * num_images):,:,:,:]
    y_valid = Y[int(valid_split * num_images):,:,:,0]
    y_valid = np.expand_dims(y_valid, axis=-1)

    return x_train, y_train, x_valid, y_valid

def data_info(X, Y):
    print("이미지의 개수:", X.shape[0])
    print("이미지의 모양:", X.shape[1],"x", X.shape[1])

    print("마스크의 개수:", Y.shape[0])
    print("마스크의 모양:", Y.shape[1],"x", Y.shape[1])
