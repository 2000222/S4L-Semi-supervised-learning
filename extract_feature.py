# -*- coding: utf-8 -*-
"""
extract the representation from saved model
"""
#import os
#import argparse
# Keras package imports
import keras
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input
# SESEMI package imports
from utils import geometric_transform, global_contrast_normalize
from utils import zca_whitener, stratified_sample, gaussian_noise
from utils import LRScheduler, DenseEvaluator, datagen, open_sesemi
from datasets import svhn, cifar10, cifar100
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras.models import Sequential, Model
from skimage.io import imsave
from sklearn.cluster import KMeans
from sklearn import manifold


#%matplotlib inline
#import itertools

def draw_features(feature_maps):
    square = 8
    ix = 1
    for _ in range(square):
        for _ in range(square):
		# specify subplot and turn of axis
        ax = plt.subplot(square, square, ix)
        ax.set_xticks([])
        ax.set_yticks([])
		# plot filter channel in grayscale
      #plt.imshow(feature_maps[0, :, :, ix-1], cmap='gray')
        plt.imshow(feature_maps[0, :, :, ix-1])
        ix += 1
      # show the figure
    return plt.show()

def main():
#load the model
    batch_size = 32
    model = keras.models.load_model('cifar10-1000-1.h5')
    print(model.summary())
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = global_contrast_normalize(x_train)
    x_test = global_contrast_normalize(x_test)
    print(x_train.shape)
    x_train = x_train.reshape((len(x_train), 32, 32, 3))
    x_test = x_test.reshape((len(x_test), 32, 32, 3))
    print(x_test.shape)
    #print(x_test[1])
    nb_classes = 10
    nb_labels = 1000
    labels_per_class = nb_labels // nb_classes
    if nb_labels == 73257:
        labels_per_class = 1000000
    sample_inds = stratified_sample(y_test, labels_per_class)
    x_labeled = x_test[sample_inds]
    y_labeled = y_test[sample_inds]
    y_labeled = to_categorical(y_labeled)
    #print(x_labeled)
    #print(y_labeled)
    print(x_labeled.shape)
    print(y_labeled.shape)
    super_datagen = ImageDataGenerator(
            width_shift_range=3,
            height_shift_range=3,
            #horizontal_flip=hflip,
            preprocessing_function=gaussian_noise,
            fill_mode='reflect',
        )
    self_datagen = ImageDataGenerator(
            width_shift_range=3,
            height_shift_range=3,
            horizontal_flip=False,
            preprocessing_function=gaussian_noise,
            fill_mode='reflect',
        )

    super_data = super_datagen.flow(
            x_labeled,y_labeled,shuffle=True, batch_size=1, seed=None)
    self_data = self_datagen.flow(
            x_test, shuffle=True, batch_size=1, seed=None)
    #super_data = x_labeled
    #self_data = x_test
    train_data_loader = datagen(super_data, self_data, batch_size)
    print('self - data')
    print(self_data) 
    print('supervised - data')
    print(super_data)
    
    print('train_data_loader')
    print(train_data_loader)
    print(len(model.layers))
    print(model.layers)
    layer_name = 'convnet_trunk'
    layer_outputs = [model.get_layer(layer_name).get_output_at(2)]
    # extract the ouputs of the top 6 layers
    activation_model = Model(inputs=model.input,outputs=layer_outputs)
    steps = len(x_test)/batch_size
    activations = activation_model.predict_generator(train_data_loader,steps=steps,verbose=0)
    print(activations)
    print(activations.shape)
    k=6666
    first_layer_activation = activations[k]
    print(first_layer_activation.shape)
    print(first_layer_activation)
    #plt.imshow(first_layer_activation)
    #plt.show()
    #imsave('first_layer_activation.jpg',first_layer_activation)
    
    
    #plt.figure(1)
    plt.matshow(first_layer_activation[:,:,0])
    plt.show()
    #imsave('first_layer_activation'+str(k)+'[:,:,0].jpg',first_layer_activation[:,:,0])
    layer_name_2 = 'self_clf'
    layer_outputs_2 = [model.get_layer(layer_name_2).get_output_at(0)]
    representation_model = Model(inputs=model.input,outputs=layer_outputs_2)
    
    final_representation = np.array(representation_model.predict_generator(train_data_loader,steps = steps,verbose = 0))
    print('The shape of final_representation')
    print(final_representation.shape)
    #print(final_representation)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    representation = scaler.fit_transform(final_representation)
    print(representation.shape)
    plot_representation = representation[:,0:2] #取其中的前两维
    print(plot_representation.shape)
    plot_Y = plot_representation
    plt.scatter(plot_Y[:, 0], plot_Y[:, 1], c = "green", marker='o', label='two')   
    #plt.show()
    plt.savefig('The_scatter_of_first_2_columns.jpg',dpi = None)
    '''kmeans = KMeans(n_clusters=10, init='k-means++')
    kmeans.fit(plot_Y)
    print(kmeans.inertia_)
    centroids = kmeans.cluster_centers_
    print(centroids)
    print(centroids.shape)'''
    #plt.scatter(centroids[:, 0], centroids[:, 1],
            #marker='x', s=169, linewidths=3,
            #color='w', zorder=10)
    #plt.savefig('centers-cifar10-sesemi-features-1.jpg')
    '''tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    X_tsne = tsne.fit_transform(X)

    print("Org data dimension is {}.Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
    plt.figure(figsize=(8, 8))
    for i in range(X_norm.shape[0]):
        plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]), 
             fontdict={'weight': 'bold', 'size': 9})
        plt.xticks([])
        plt.yticks([])
        plt.show()
        plt.savefig('t-SNE-cifar10.jpg')'''
    '''superd = next(super_data)
    selfd = next(self_data)
    print('super - d')
    print(superd)
    print(superd.shape())
    print('self - d')
    print(selfd)    

    print('Try to extract the representation of the sesemi model')
    fig = plt.figure(figsize=(14,10))
    for n in range(1,29):
        fig.add_subplot(4, 7, n)
        img_tensor = [self_data[n],super_data[n]]
        #img_tensor = np.expand_dims(img_tensor, axis=0)
        #img_tensor /= 255.
        print('image tensor to be shown')
        print(img_tensor)
        print(len(img_tensor))
        #plt.imshow(self_data)
        #plt.show()
        #print(img_tensor2.shape)
        #img = expand_dims(img, axis=0)i
        #img = preprocess_input(img)
        img_tensor = list(itertools.chain.from_iterable(img_tensor))
        print(img_tensor.shape())
        img_tensor.flatten()
        print(img_tensor)
        feature_maps = model.predict(img_tensor)
        print(feature_maps)
        draw_features(feature_maps)
        plt.axis('off')
        plt.show()
    return
    print('Try to visualize the representation!')'''
    return

if __name__ == '__main__':
    main()