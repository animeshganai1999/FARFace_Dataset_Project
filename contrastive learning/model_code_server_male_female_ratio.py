import os
import cv2
import numpy as np
import pandas as pd
import GenderModel
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.callbacks import EarlyStopping
import random


# Use only GPU-0
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


target_shape = (224,224)

# Function to read and preprocess the image
def read_and_preprocess_image(filepath):
    img = cv2.imread(filepath)
    img = cv2.resize(img, (224, 224))  # Resize to match VGG Face input shape
    img = img / 255.0  # Normalize pixel values to [0, 1]
    img = img.astype(np.float32)
    return img

print("Tarining data Loading begin")
############################################################# TARINING DATA #############################################################

# Define the paths to the anchor, positive, and negative folders
anchor_folder = 'train_val_data/tarining_data/anchor'
positive_folder = 'train_val_data/tarining_data/positive'
negative_folder = 'train_val_data/tarining_data/negative'

# Get the filenames in each folder and sort them based on their integer values
len_images = len(os.listdir(anchor_folder))
images_sorted = []
for i in range(len_images):
    images_sorted.append(str(i)+'.jpg')
    
anchor_filenames = positive_filenames = negative_filenames = images_sorted

# Load the label of each data
df_gender_train = pd.read_csv('train_val_data/train_gender.csv')
df_gender_train['gender'] = df_gender_train['gender'].replace({'M': 1, 'F': 0})
y_gender_train_old = np.array(list(df_gender_train['gender']))

## Get female indices
female_indices = y_gender_train_old == 0
male_indices_all = y_gender_train_old == 1
num_female = len(df_gender_train[df_gender_train['gender'] == 0])
num_male = 2*num_female

print(num_male,num_female)

female_indices = list(np.where(female_indices)[0])
male_indices_all = list(np.where(male_indices_all)[0])

male_indices = random.sample(male_indices_all,num_male)

# Create triplets (anchor, positive, negative)
triplets = []
## Loading female images
for i in female_indices:
    anchor_filepath = anchor_folder+'/'+str(i)+'.jpg'
    positive_filepath = positive_folder+'/'+str(i)+'.jpg'
    negative_filepath = negative_folder+'/'+str(i)+'.jpg'
    
    anchor_image = read_and_preprocess_image(anchor_filepath)
    positive_image = read_and_preprocess_image(positive_filepath)
    negative_image = read_and_preprocess_image(negative_filepath)
    gender_label = y_gender_train_old[i]
    
    triplets.append((anchor_image, positive_image, negative_image,gender_label))

## Loading male images
for i in male_indices:
    anchor_filepath = anchor_folder+'/'+str(i)+'.jpg'
    positive_filepath = positive_folder+'/'+str(i)+'.jpg'
    negative_filepath = negative_folder+'/'+str(i)+'.jpg'
    
    anchor_image = read_and_preprocess_image(anchor_filepath)
    positive_image = read_and_preprocess_image(positive_filepath)
    negative_image = read_and_preprocess_image(negative_filepath)
    gender_label = y_gender_train_old[i]
    
    triplets.append((anchor_image, positive_image, negative_image,gender_label))

## randomly shuffle dataset
random.shuffle(triplets)
# Convert triplets to numpy arrays
anchor_images = np.array([t[0] for t in triplets])
positive_images = np.array([t[1] for t in triplets])
negative_images = np.array([t[2] for t in triplets])
y_gender_train_old = np.array([t[3] for t in triplets])

n = y_gender_train_old.shape[0]
y_gender_train = np.zeros((n, 2), dtype=int)
# Use boolean indexing to set (i, 0) or (i, 1) index to 1 based on the original array
y_gender_train[np.arange(n), y_gender_train_old] = 1

x_train = [anchor_images,positive_images,negative_images]

print("Tarining data Loading End")
##########################################################################################################################

############################################################# VALIDATION DATA #############################################################
print("Validation data Loading begin")
# Define the paths to the anchor, positive, and negative folders
anchor_folder_val = 'train_val_data/validation_data/anchor'
positive_folder_val = 'train_val_data/validation_data/positive'
negative_folder_val = 'train_val_data/validation_data/negative'

# Get the filenames in each folder and sort them based on their integer values
len_images = len(os.listdir(anchor_folder_val))
images_sorted = []
for i in range(len_images):
    images_sorted.append(str(i)+'.jpg')
    
anchor_filenames_val = positive_filenames_val = negative_filenames_val = images_sorted

# Load the label of each data
df_gender_train = pd.read_csv('train_val_data/val_gender.csv')
df_gender_train['gender'] = df_gender_train['gender'].replace({'M': 1, 'F': 0})
y_gender_train_old_val = np.array(list(df_gender_train['gender']))

## Get female indices
female_indices = y_gender_train_old_val == 0
male_indices_all = y_gender_train_old_val == 1
num_female = len(df_gender_train[df_gender_train['gender'] == 0])
num_male = 2*num_female

female_indices = list(np.where(female_indices)[0])
male_indices_all = list(np.where(male_indices_all)[0])

male_indices = random.sample(male_indices_all,num_male)

# Create triplets (anchor, positive, negative)
triplets_val = []
## Loading female images
for i in female_indices:
    anchor_filepath = anchor_folder_val+'/'+str(i)+'.jpg'
    positive_filepath = positive_folder_val+'/'+str(i)+'.jpg'
    negative_filepath = negative_folder_val+'/'+str(i)+'.jpg'
    
    anchor_image = read_and_preprocess_image(anchor_filepath)
    positive_image = read_and_preprocess_image(positive_filepath)
    negative_image = read_and_preprocess_image(negative_filepath)
    gender_label = y_gender_train_old_val[i]
    
    triplets_val.append((anchor_image, positive_image, negative_image,gender_label))

## Loading male images
for i in male_indices:
    anchor_filepath = anchor_folder_val+'/'+str(i)+'.jpg'
    positive_filepath = positive_folder_val+'/'+str(i)+'.jpg'
    negative_filepath = negative_folder_val+'/'+str(i)+'.jpg'
    
    anchor_image = read_and_preprocess_image(anchor_filepath)
    positive_image = read_and_preprocess_image(positive_filepath)
    negative_image = read_and_preprocess_image(negative_filepath)
    gender_label = y_gender_train_old_val[i]
    
    triplets_val.append((anchor_image, positive_image, negative_image,gender_label))

## randomly shuffle dataset
random.shuffle(triplets_val)
# Convert triplets to numpy arrays
anchor_images_val = np.array([t[0] for t in triplets_val])
positive_images_val = np.array([t[1] for t in triplets_val])
negative_images_val = np.array([t[2] for t in triplets_val])
y_gender_val_old = np.array([t[3] for t in triplets_val])

n = y_gender_val_old.shape[0]
y_gender_val = np.zeros((n, 2), dtype=int)
# Use boolean indexing to set (i, 0) or (i, 1) index to 1 based on the original array
y_gender_val[np.arange(n), y_gender_val_old] = 1

x_val = [anchor_images_val,positive_images_val,negative_images_val]

print("Validation data Loading end")
##########################################################################################################################


## Load the model
loaded_model = GenderModel.loadModel()
for layer in loaded_model.layers:
        layer.trainable = True  # Set all layers to trainable

# Define separate input layers for anchor, positive, and negative images
input_anchor = Input(shape=(224, 224, 3), name='input_anchor')
input_positive = Input(shape=(224, 224, 3), name='input_positive')
input_negative = Input(shape=(224, 224, 3), name='input_negative')

# Get the desired layer 'conv2d_12' from the loaded model
desired_layer = loaded_model.get_layer('conv2d_12')

# Create a new model that goes from the input layer to the desired layer 'conv2d_12'
embedding_model = Model(inputs=loaded_model.input, outputs=desired_layer.output)
# Use the embedding_model to obtain the embeddings from 'conv2d_12' for anchor, positive, and negative images
anchor_embeddings = embedding_model(input_anchor)
positive_embeddings = embedding_model(input_positive)
negative_embeddings = embedding_model(input_negative)

# Concatenate the embeddings from anchor, positive, and negative images
combined_embeddings = Concatenate(axis=-1)([anchor_embeddings, positive_embeddings, negative_embeddings])

# Create the combined model with the three input layers and the two outputs
model = Model(inputs=[input_anchor, input_positive, input_negative], outputs=[combined_embeddings,loaded_model(input_anchor)])


# Define the triplet loss function
def triplet_loss(y_true, y_pred, alpha=0.2):
    anchor, positive, negative = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]
    pos_dist = K.sum(K.square(anchor - positive), axis=-1)
    neg_dist = K.sum(K.square(anchor - negative), axis=-1)
    basic_loss = pos_dist - neg_dist + alpha
    loss = K.maximum(basic_loss, 0.0)
    return loss

losses = {
    'concatenate': triplet_loss,  # Output layer name for embeddings
    'model': BinaryCrossentropy()  # Output layer name for gender prediction
}
# Compile the model with triplet loss and binary cross-entropy loss for gender prediction
model.compile(
        optimizer=Adam(learning_rate=0.00001),
        loss=losses,
        loss_weights={'concatenate': 0.7, 'model': 0.3},  # You can adjust the loss weights as needed
        metrics={'model': 'accuracy'}
    )


# # Custom training loop for triplet loss and gender prediction
def custom_generator(x_data, y_gender_data, anchor_indices, positive_indices, negative_indices, batch_size=32):
    while True:
        num_samples = len(anchor_indices)
        indices = list(range(num_samples))
        random.shuffle(indices)
        start_index = 0
        while start_index < num_samples:
            batch_indices = indices[start_index : start_index + batch_size]
            # Extract the anchor, positive, and negative examples from the corresponding lists
            anchor_batch = x_data[0][anchor_indices[batch_indices]]
            positive_batch = x_data[1][anchor_indices[batch_indices]]
            negative_batch = x_data[2][anchor_indices[batch_indices]]

            # Extract the gender labels for the anchor examples
            gender_label_batch = y_gender_data[batch_indices]
            yield [anchor_batch, positive_batch, negative_batch], [anchor_batch, gender_label_batch]

            start_index += batch_size


# Create indices for the anchor, positive, and negative images
num_samples = len(anchor_images)
indices = np.arange(num_samples)
anchor_indices = positive_indices = negative_indices = indices

### For validation set #######  
anchor_indices_val = positive_indices_val = negative_indices_val = np.arange(len(y_gender_val))

# Train the model using the custom generator
batch_size = 16
num_steps_per_epoch = len(anchor_indices) // batch_size
epochs = 10

val_steps_per_epoch = len(anchor_indices_val) // batch_size

early_stopping = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
model.fit(custom_generator(x_train, y_gender_train, anchor_indices, positive_indices, negative_indices, batch_size),
          steps_per_epoch=num_steps_per_epoch,
          epochs=epochs,
          validation_data = custom_generator(x_val, y_gender_val, anchor_indices_val, positive_indices_val, negative_indices_val, batch_size),
          validation_steps=val_steps_per_epoch,
          callbacks = [early_stopping]
)

model.save('vgg_face_triplet_loss_full_mf_'+str(epochs)+'_epochs.h5')


