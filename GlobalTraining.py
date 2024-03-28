# %%
import tensorflow as tf
print(tf.__version__)
import numpy as np
import keras.backend as K
import os
import random
import matplotlib.pyplot as plt
tfk = tf.keras
tfkl = tf.keras.layers
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns

# Random seed for reproducibility
seed = 21
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
tf.compat.v1.set_random_seed(seed)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("Versione di TensorFlow:", tf.__version__)

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(physical_devices[0], 'GPU')

# %%
#Parameters

path_images = "/dataset/new_simones/AerialWaste2"
path_json = "/home/simones95/EnvCrime"
AW1 = False
AW2 = not AW1

oversampling = True

batch_size = 8
spatial_resolution = 20
max_dim = 1088 
max_shape = tf.constant((max_dim, max_dim))

learning_rate = 1.0e-5
l2 = 1.0e-1

top_t = 0.01 
l1 = 1.0

# %%
def load_data(name_json):
    # Load dataset for binary classification
    dataset = json.load(open(os.path.join(path_json, name_json)))
    if 'training' in name_json:
        pairs = [[os.path.join(path_images, 'images', d['file_name']), d['is_candidate_location']] for d in dataset['images']]
    else:
        if AW1: 
            pairs = [[os.path.join(path_images, 'images', d['file_name']), d['is_candidate_location']] for d in dataset['images']]
        else:
            pairs = [[os.path.join(path_images, 'images', d['file_name']), d['is_suspicius_location']] for d in dataset['images']]
            
    #Shuffle the pairs
    random.shuffle(pairs)
    
    # Extract info_images and their binary labels
    info_images, binary_labels = zip(*pairs)
    info_images = list(info_images)
    binary_labels = list(binary_labels)
    
    return info_images, binary_labels

if AW1:
    training_json = "training_aw1.json"
    testing_json = "testing_aw1.json"
elif AW2:
    training_json = "training_aw2.json"
    testing_json = "testing_aw2.json"

training_info_images, training_binary_labels = load_data(training_json)
num_samples_training = int(len(training_info_images) * 0.8)
validation_info_images, validation_binary_labels = training_info_images[num_samples_training:], training_binary_labels[num_samples_training:]
training_info_images, training_binary_labels = training_info_images[:num_samples_training], training_binary_labels[:num_samples_training]
testing_info_images, testing_binary_labels = load_data(testing_json)

print("Training: ", len(training_info_images), " ", len(training_info_images)/10434)
print("Validation: ", len(validation_info_images), " ", len(validation_info_images)/10434)
print("Testing: ", len(testing_info_images), " ", len(testing_info_images)/10434)

print("Training - Positive: ", sum(training_binary_labels), " - Negative: ", len(training_binary_labels) - sum(training_binary_labels))
print("Validation - Positive: ", sum(validation_binary_labels), " - Negative: ", len(validation_binary_labels) - sum(validation_binary_labels))
print("Testing - Positive: ", sum(testing_binary_labels), " - Negative: ", len(testing_binary_labels) - sum(testing_binary_labels))

# %%
def RandomOversampling(info_images, binary_labels):
    upsampling_factor = len(binary_labels)-2*sum(binary_labels)
    positive_indices = [i for i, value in enumerate(binary_labels) if value == 1]
    random.shuffle(positive_indices)
    for i in range(upsampling_factor):
        if i < len(positive_indices):
            info_images.append(info_images[positive_indices[i]])
            binary_labels.append(binary_labels[positive_indices[i]])
        
    upsampling_factor = len(binary_labels)-2*sum(binary_labels)
    if upsampling_factor > 0:
        info_images, binary_labels = RandomOversampling(info_images, binary_labels)
        
    random.seed(seed)
    random.shuffle(info_images)
    random.seed(seed)
    random.shuffle(binary_labels)
    
    return info_images, binary_labels

if oversampling:
    training_info_images, training_binary_labels = RandomOversampling(training_info_images, training_binary_labels)

print("Training - Positive: ", sum(training_binary_labels), " - Negative: ", len(training_binary_labels) - sum(training_binary_labels))

# %%
for index in range(len(training_binary_labels)):
    training_binary_labels[index] = [training_binary_labels[index]]*4 + [0]
for index in range(len(validation_binary_labels)):
    validation_binary_labels[index] = [validation_binary_labels[index]]*4 + [0]
for index in range(len(testing_binary_labels)):
    testing_binary_labels[index] = [testing_binary_labels[index]]*4 + [0]

# %%
# Custom Generator
def load(info_image, label):
    image = tf.io.read_file(info_image)
    image = tf.image.decode_png(image, channels=3)
    #image = tf.image.rgb_to_grayscale(image)
    
    # Resize to a specific resolution
    shape = tf.shape(image)
    spatial_resolution_image = 210 * 100 / tf.maximum(shape[0],shape[1])
    new_shape_image = tf.cast(tf.cast(shape, tf.float64) * (spatial_resolution_image / spatial_resolution), tf.int32)
    image = tf.image.resize(image, size=new_shape_image[0:2])
    
    # Padding
    offset = max_shape - new_shape_image[0:2]
    image = tf.image.pad_to_bounding_box(image, offset[0]//2, offset[1]//2, max_dim, max_dim)
    
    return image, {'p1':label[0],'p2':label[0],'p3':label[0],'p4':label[0],'p':label[0:4],
                   'sparse1':label[4],'sparse2':label[4],'sparse3':label[4],'sparse4':label[4]}

def make_dataset(info_images, binary_labels):
    return tf.data.Dataset.from_tensor_slices((info_images, binary_labels))\
        .map(load)\
        .batch(batch_size)\
        .prefetch(tf.data.AUTOTUNE)

train_ds = make_dataset(training_info_images, training_binary_labels)
val_ds = make_dataset(validation_info_images, validation_binary_labels)
test_ds = make_dataset(testing_info_images, testing_binary_labels)

# %%
for i,o in train_ds.take(1):
  print(i.shape)
  print(o)
  plt.figure(figsize=(20, 20))
  for index in range(2):
    plt.subplot(2, 4, index+1)
    plt.imshow(i[index]/255, cmap='gray')
    plt.axis('off')

  plt.show()

# %%
class RandomNoiseLayer(tf.keras.layers.Layer):
    def __init__(self, noise_factor, **kwargs):
        super(RandomNoiseLayer, self).__init__(**kwargs)
        self.noise_factor = noise_factor

    def call(self, inputs, training=None):
        if training:
            noisy_inputs = inputs + self.noise_factor * tf.random.normal(shape=tf.shape(inputs))
            noisy_inputs = tf.clip_by_value(noisy_inputs, 0.0, 255.0)
            return noisy_inputs
        return inputs

    def get_config(self):
        config = super().get_config()
        config.update({'noise_factor': self.noise_factor})
        return config


# %%
class TopTPooling(tf.keras.layers.Layer):
    def __init__(self, t, **kwargs):
        super(TopTPooling, self).__init__(**kwargs)
        self.t = t
        
    def call(self, inputs, training=None):
        shape = tf.shape(inputs)
        
        n_values = tf.cast(tf.reduce_prod(shape[1:3]), tf.float32)
        n_top_values = tf.cast(self.t * n_values, tf.int32)
        
        x = tf.reshape(inputs, shape=(tf.cast(tf.reduce_prod(shape[0]), tf.int32), tf.cast(n_values, tf.int32),tf.cast(tf.reduce_prod(shape[3]), tf.int32)))
        x = tf.sort(x, direction='DESCENDING', axis=1)
        x = tf.reduce_mean(x[:, :n_top_values], axis=1)
        x = tf.reshape(x, shape=(tf.cast(tf.reduce_prod(shape[0]), tf.int32), tf.cast(tf.reduce_prod(shape[3]), tf.int32)))
        
        return x

    def get_config(self):
        config = super().get_config()
        config.update({'t': self.t})
        return config
    
#t = tf.random.uniform(shape=(batch_size,4,4,2))


# %%
class SparsityLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SparsityLayer, self).__init__(**kwargs)

    def call(self, inputs):
        
        shape = tf.shape(inputs)
        
        sparsity = tf.reduce_sum(inputs, axis=(1,2,3))/tf.cast(tf.reduce_prod(shape[1:3]), tf.float32)
        sparsity = tf.reshape(sparsity, shape=(tf.cast(tf.reduce_prod(shape[0]), tf.int32), tf.cast(tf.reduce_prod(shape[3]), tf.int32)))
        
        return sparsity

    def get_config(self):
        config = super(SparsityLayer, self).get_config()
        return config
    
# %%
class ResizeLayer(tf.keras.layers.Layer):
    def __init__(self, s, **kwargs):
        super(ResizeLayer, self).__init__(**kwargs)
        self.s = s

    def call(self, inputs):

        return tf.image.resize(inputs, size=(self.s,self.s), method="bicubic")

    def get_config(self):
        config = super().get_config()
        config.update({'s': self.s})
        return config

# %%
# Model

input = tfkl.Input(shape=(max_dim, max_dim, 3), name='input')
flipped_image = tfkl.RandomFlip(name='random_flip')(input)
rotated_image = tfkl.RandomRotation(1, fill_mode='constant', fill_value=0.0, name='random_rotation')(flipped_image)
noised_image = RandomNoiseLayer(20.0, name='random_noise')(rotated_image)
brightened_image = tfkl.RandomBrightness(0.4, name='random_brightness')(noised_image)
contrasted_image = tfkl.RandomContrast(0.4, name='random_constrast')(brightened_image)
preprocessed_image = tf.keras.applications.resnet50.preprocess_input(contrasted_image)
backbone = tfk.applications.resnet50.ResNet50(include_top=False, weights='imagenet', input_tensor=preprocessed_image, pooling=None)

c1 = backbone.get_layer(name='conv2_block3_out').output
c2 = backbone.get_layer(name='conv3_block4_out').output
c3 = backbone.get_layer(name='conv4_block6_out').output
c4 = backbone.get_layer(name='conv5_block3_out').output

c1 = tfkl.Conv2D(filters=128, kernel_size=(1,1), padding='same', activation='relu', kernel_regularizer=tfk.regularizers.l2(l2), name="p11")(c1)
c1 = tfkl.Conv2D(filters=64, kernel_size=(1,1), padding='same', activation='relu', kernel_regularizer=tfk.regularizers.l2(l2), name="p12")(c1)
c1 = tfkl.Conv2D(filters=32, kernel_size=(1,1), padding='same', activation='relu', kernel_regularizer=tfk.regularizers.l2(l2), name="p13")(c1)
c1 = tfkl.Conv2D(filters=1, kernel_size=(1,1), padding='same', activation='sigmoid', kernel_regularizer=tfk.regularizers.l2(l2), name="p14")(c1)

c2 = tfkl.Conv2D(filters=128, kernel_size=(1,1), padding='same', activation='relu', kernel_regularizer=tfk.regularizers.l2(l2), name="p21")(c2)
c2 = tfkl.Conv2D(filters=64, kernel_size=(1,1), padding='same', activation='relu', kernel_regularizer=tfk.regularizers.l2(l2), name="p22")(c2)
c2 = tfkl.Conv2D(filters=32, kernel_size=(1,1), padding='same', activation='relu', kernel_regularizer=tfk.regularizers.l2(l2), name="p23")(c2)
c2 = tfkl.Conv2D(filters=1, kernel_size=(1,1), padding='same', activation='sigmoid', kernel_regularizer=tfk.regularizers.l2(l2), name="p24")(c2)

c3 = tfkl.Conv2D(filters=128, kernel_size=(1,1), padding='same', activation='relu', kernel_regularizer=tfk.regularizers.l2(l2), name="p31")(c3)
c3 = tfkl.Conv2D(filters=64, kernel_size=(1,1), padding='same', activation='relu', kernel_regularizer=tfk.regularizers.l2(l2), name="p32")(c3)
c3 = tfkl.Conv2D(filters=32, kernel_size=(1,1), padding='same', activation='relu', kernel_regularizer=tfk.regularizers.l2(l2), name="p33")(c3)
c3 = tfkl.Conv2D(filters=1, kernel_size=(1,1), padding='same', activation='sigmoid', kernel_regularizer=tfk.regularizers.l2(l2), name="p34")(c3)

c4 = tfkl.Conv2D(filters=128, kernel_size=(1,1), padding='same', activation='relu', kernel_regularizer=tfk.regularizers.l2(l2), name="p41")(c4)
c4 = tfkl.Conv2D(filters=64, kernel_size=(1,1), padding='same', activation='relu', kernel_regularizer=tfk.regularizers.l2(l2), name="p42")(c4)
c4 = tfkl.Conv2D(filters=32, kernel_size=(1,1), padding='same', activation='relu', kernel_regularizer=tfk.regularizers.l2(l2), name="p43")(c4)
c4 = tfkl.Conv2D(filters=1, kernel_size=(1,1), padding='same', activation='sigmoid', kernel_regularizer=tfk.regularizers.l2(l2), name="p44")(c4)

p4 = TopTPooling(top_t, name='p4')(c4)
p3 = TopTPooling(top_t, name='p3')(c3)
p2 = TopTPooling(top_t, name='p2')(c2)
p1 = TopTPooling(top_t, name='p1')(c1)

#c1 = ResizeLayer(max_dim)(c1)
#c2 = ResizeLayer(max_dim)(c2)
#c3 = ResizeLayer(max_dim)(c3)
#c4 = ResizeLayer(max_dim)(c4)

sparse1 = SparsityLayer(name="sparse1")(c1)
sparse2 = SparsityLayer(name="sparse2")(c2)
sparse3 = SparsityLayer(name="sparse3")(c3)
sparse4 = SparsityLayer(name="sparse4")(c4)

p = tfkl.Concatenate(name='p')([p1,p2,p3,p4])

binary_classifier = tfk.Model(input, [p1,p2,p3,p4,p,sparse1,sparse2,sparse3,sparse4], name="binary_classifier")

binary_classifier.summary()

# %%
# Create a learning rate scheduler callback.
lr_cb = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_p_binary_accuracy", mode="max", factor=0.25, patience=5)
# Create an early stopping callback.
es_cb = tf.keras.callbacks.EarlyStopping(monitor="val_p_binary_accuracy", mode="max", patience=8, restore_best_weights=True)

# Compile the model
binary_classifier.compile(
    # Optimizer
    optimizer=tfk.optimizers.Adam(learning_rate=learning_rate), 
    # Loss function to minimize
    loss={'p1':'binary_crossentropy','p2':'binary_crossentropy',
          'p3':'binary_crossentropy','p4':'binary_crossentropy',
          'sparse1':'mean_absolute_error','sparse2':'mean_absolute_error',
          'sparse3':'mean_absolute_error','sparse4':'mean_absolute_error'},
    loss_weights={'p1':1.0,'p2':1.0,'p3':1.0,'p4':1.0,
                  'sparse1':l1,'sparse2':l1,'sparse3':l1,'sparse4':l1},
    # List of metrics to monitor
    metrics={'p1':[tfk.metrics.BinaryAccuracy(),tfk.metrics.Precision(),tfk.metrics.Recall()],
             'p2':[tfk.metrics.BinaryAccuracy(),tfk.metrics.Precision(),tfk.metrics.Recall()],
             'p3':[tfk.metrics.BinaryAccuracy(),tfk.metrics.Precision(),tfk.metrics.Recall()],
             'p4':[tfk.metrics.BinaryAccuracy(),tfk.metrics.Precision(),tfk.metrics.Recall()],
             'p':[tfk.metrics.BinaryAccuracy(),tfk.metrics.Precision(),tfk.metrics.Recall()]
    }
)

# Train the model
history = binary_classifier.fit(
    train_ds,
    epochs=200,
    validation_data=val_ds,
    callbacks=[lr_cb, es_cb],
)

binary_classifier.evaluate(test_ds, verbose=2)

binary_classifier.save("/home/simones95/EnvCrime/Prova.h5")


