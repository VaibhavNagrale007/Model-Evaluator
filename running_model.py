import os
import threading
import numpy as np
import foolbox as fb
import tensorflow as tf
import tensorflow_addons as tfa
from PIL import Image
from vit_keras import vit
from flask import request
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# BATCH_SIZE = 32 
IMG_SIZE = (224, 224)

def get_loader_data(data_loader):   # For getting images and labels from dataset_generator
    for images, labels in data_loader:
        images = images
        labels = labels
        break
    return images, labels

def generator(path_to_folder, batch_size):      # For getting data_generator
    dataset_datagen = ImageDataGenerator(rescale=1./255)
    dataset_generator = dataset_datagen.flow_from_directory(
        path_to_folder,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode='binary',
        color_mode='rgb'
    )
    print('Image shape:',dataset_generator.image_shape)
    print("Number of images in dataset_generator:", dataset_generator.n)
    return dataset_generator

def evaluate_dataset(model, dataset_path, accuracy):   # For evaluation on default dataset
    print('Defualt dataset')
    dataset_generator = generator(dataset_path, 32)
    dataset_loss, dataset_accuracy = model.evaluate(dataset_generator)
    print("Dataset Loss:", dataset_loss)
    print("Dataset Accuracy:", dataset_accuracy)
    accuracy.update({"dataset":[round(dataset_accuracy, 3), round(dataset_loss, 3)]})

def run_fgsm_attack(model, fgsm_attack_dataset, accuracy):    # For evaluation on fgsm attack
    print('FGSM Attack')
    dataset_generator = generator(fgsm_attack_dataset, 16)    # keep batch size less for faster since it chooses only first batch
    loaded_model = model

    # Check if the previous layer is a Dense layer with 1 unit
    if isinstance(loaded_model.layers[-1], Dense) and loaded_model.layers[-1].output_shape[-1] == 1:
        x = loaded_model.layers[-2].output
        new_output = Dense(2, activation='sigmoid', name='new_output')(x)
        modified_model = Model(inputs=loaded_model.input, outputs=new_output)
        modified_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    else:
        modified_model = loaded_model
    
    preprocessing = dict(flip_axis=-1, mean=[103.939, 116.779, 123.68])
    bounds = (0, 255)
    fmodel = fb.TensorFlowModel(modified_model, bounds=bounds, preprocessing=preprocessing)
    fmodel = fmodel.transform_bounds((0, 1))
    assert fmodel.bounds == (0, 1)
    
    images, labels = get_loader_data(dataset_generator)
    images_tf = tf.constant(images)
    labels_tf = tf.constant(labels)
    attack = fb.attacks.LinfDeepFoolAttack()
    # epsilons = np.linspace(0.03, 1, num=20)
    epsilon = 0.05

    raw, clipped, is_adv = attack(fmodel, images_tf, labels_tf, epsilons=epsilon)
    is_adv_float32 = tf.cast(is_adv, tf.float32)
    mean_adv = tf.reduce_mean(is_adv_float32, axis=-1)
    robust_accuracy = 1 - mean_adv
    print("FGSM Attack Dataset Accuracy:", robust_accuracy.numpy())

    accuracy.update({"fgsm":[round(robust_accuracy.numpy(), 3), 'NA (epsilon = 0.05)']})

def run_red_attack(model, red_attack_dataset,accuracy):     # For evaluation on RED attack
    print('RED Attack')
    red_attack_dataset_generator = generator(red_attack_dataset, 32)
    print(red_attack_dataset_generator.image_shape)
    print("Number of images in red_attack_dataset_generator:", red_attack_dataset_generator.n)

    # Evaluate on dataset
    red_attack_dataset_loss, red_attack_dataset_accuracy = model.evaluate(red_attack_dataset_generator)
    print("RED Attack Dataset Loss:", red_attack_dataset_loss)
    print("RED Attack Dataset Accuracy:", red_attack_dataset_accuracy)

    accuracy.update({"red":[round(red_attack_dataset_accuracy, 3), round(red_attack_dataset_loss, 3)]})

def run_model_on_dataset(model_path):
    # Load the model from the specified folder
    model = tf.keras.models.load_model(model_path)

    # Directory paths
    dataset = '/home/vaibhavbnagrale/Downloads/btp/Datasets/xray_pneumonia/val'
    red_attack_dataset = '/home/vaibhavbnagrale/Downloads/btp/Datasets/xray_pneumonia/val'

    accuracy = {}   # For storing accuracy of each attack performed

    threads = []    # Applied threading for evaluating model on different attacks parallelly
    t1 = threading.Thread(target=evaluate_dataset, args=(model, dataset, accuracy))       # Thread 1: Evaluate dataset
    threads.append(t1)
    
    if request.form.get('fgsm'):
        t2 = threading.Thread(target=run_fgsm_attack, args=(model, dataset, accuracy))    # Thread 2: Run FGSM attack
        threads.append(t2)
    
    if request.form.get('red'):
        t3 = threading.Thread(target=run_red_attack, args=(model, red_attack_dataset, accuracy))    # Thread 3: Run RED attack 
        threads.append(t3)

    for t in threads:   # Start all threads
        t.start()

    for t in threads:   # Wait for all threads to complete
        t.join()
        
    return accuracy
