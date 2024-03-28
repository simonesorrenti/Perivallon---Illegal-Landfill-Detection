# %%
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

batch_size = 10
spatial_resolution = 50
max_dim = 512
max_shape = tf.constant((max_dim, max_dim))

learning_rate = 1.0e-5
l2 = 1.0e-2

hide_size = 16
hide_probability = 0.0

patch_size = 32
L = 256

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
    
    return image, label

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
class UnrollPatchesLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(UnrollPatchesLayer, self).__init__(**kwargs)

    def call(self, inputs):
        
        shape = tf.shape(inputs)

        output = tf.reshape(inputs, shape=(tf.cast(tf.reduce_prod(shape[0:2]), tf.int32), tf.cast(tf.reduce_prod(shape[2]), tf.int32),
                                              tf.cast(tf.reduce_prod(shape[3]), tf.int32), tf.cast(tf.reduce_prod(shape[4]), tf.int32)))
        
        return output

    def get_config(self):
        config = super(UnrollPatchesLayer, self).get_config()
        return config

# %%
class AggregationLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AggregationLayer, self).__init__(**kwargs)

    def call(self, inputs):

        shape = tf.shape(inputs[1])
        shape2 = tf.shape(inputs[0])

        output = tf.reshape(inputs[0], shape=(tf.cast(tf.reduce_prod(shape[0]), tf.int32), tf.cast(tf.reduce_prod(shape[1]), tf.int32)))

        return output

    def get_config(self):
        config = super(AggregationLayer, self).get_config()
        return config
    
# %%
class ExtractPatchesLayer(tf.keras.layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super(ExtractPatchesLayer, self).__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, inputs):

        patches = tf.image.extract_patches(images=inputs, sizes=[1, self.patch_size, self.patch_size, 1], strides=[1, self.patch_size, self.patch_size, 1], rates=[1, 1, 1, 1], padding='VALID')
        
        shape = tf.shape(patches)
        patches = tf.reshape(
            patches,
            shape=(shape[0]*shape[1]*shape[2], self.patch_size, self.patch_size, 3)
        )

        return patches

    def get_config(self):
        config = super(ExtractPatchesLayer, self).get_config()
        config.update({'patch_size': self.patch_size})
        return config
    
# %%
# %%
class GAMLayer(tf.keras.layers.Layer):
    def __init__(self, L, **kwargs):
        super(GAMLayer, self).__init__(**kwargs)
        self.L = L
        self.V_dense = tfkl.Dense(units=self.L, activation='tanh', kernel_regularizer=tfk.regularizers.l2(l2), name="V")
        self.U_dense = tfkl.Dense(units=self.L, activation='sigmoid', kernel_regularizer=tfk.regularizers.l2(l2), name="U")
        self.w_dense = tfkl.Dense(units=1, activation='linear', kernel_regularizer=tfk.regularizers.l2(l2), name="w")

    def call(self, inputs):

        V = self.V_dense(inputs)
        U = self.U_dense(inputs)
        VxU = tf.math.multiply(V, U, name='VxU')
        w = self.w_dense(VxU)

        return w

    def get_config(self):
        config = super(GAMLayer, self).get_config()
        config.update({'L': self.L})
        return config
    
# %%
class AggregateFeaturesLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AggregateFeaturesLayer, self).__init__(**kwargs)

    def call(self, inputs):

        w_shape = tf.shape(inputs[1])
        gap_shape = tf.shape(inputs[0])

        w = tf.reshape(inputs[1], shape=(w_shape[0], w_shape[1], 1))
        gaps = tf.reshape(inputs[0], shape=(w_shape[0], w_shape[1], gap_shape[-1]))

        feature_vector = tf.multiply(gaps,w)
        feature_vector = tf.reduce_sum(feature_vector, axis=1)

        return feature_vector

    def get_config(self):
        config = super(AggregateFeaturesLayer, self).get_config()
        return config
    
# %%
class MILInputLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(MILInputLayer, self).__init__(**kwargs)

    def call(self, inputs):

        return inputs

    def get_config(self):
        config = super(MILInputLayer, self).get_config()
        return config

# %%
class ReshapeWeightsLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ReshapeWeightsLayer, self).__init__(**kwargs)

    def call(self, inputs):

        input_shape = tf.shape(inputs[1])
        w_shape = tf.shape(inputs[0])

        w = tf.reshape(
            inputs[0],
            shape=(input_shape[0], w_shape[0]//input_shape[0])
        )

        return w

    def get_config(self):
        config = super(ReshapeWeightsLayer, self).get_config()
        return config

# %%
class FeaturesMapsLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(FeaturesMapsLayer, self).__init__(**kwargs)

    def call(self, inputs):

        return inputs

    def get_config(self):
        config = super(FeaturesMapsLayer, self).get_config()
        return config

# %% 
    
class HideLayer(tf.keras.layers.Layer):
    def __init__(self, hide_size, hide_probability, **kwargs):
        super(HideLayer, self).__init__(**kwargs)
        self.hide_size = hide_size
        self.hide_probability = hide_probability

    def call(self, inputs, training=None):
        if training:
          shape = tf.shape(inputs)

          mask = tf.random.uniform(shape=(shape[0],shape[1]//self.hide_size,shape[2]//self.hide_size,1), dtype=tf.float32)
          mask = tf.where(mask > self.hide_probability, 1.0, 0.0)

          mask = tf.repeat(tf.repeat(mask, repeats=self.hide_size, axis=1), repeats=self.hide_size, axis=2)
          mask = tf.tile(mask, [1, 1, 1, 3])
          inputs = tf.multiply(inputs,mask)
        return inputs

    def get_config(self):
        config = super().get_config()
        config.update({'hide_size': self.hide_size})
        config.update({'hide_probability': self.hide_probability})
        return config

# %%
# Model

input = tfkl.Input(shape=(None, None, 3), name='input')
hide_image = HideLayer(hide_size, hide_probability)(input)
flipped_image = tfkl.RandomFlip(name='random_flip')(hide_image)
rotated_image = tfkl.RandomRotation(1, fill_mode='constant', fill_value=0.0, name='random_rotation')(flipped_image)
noised_image = RandomNoiseLayer(20.0, name='random_noise')(rotated_image)
brightened_image = tfkl.RandomBrightness(0.4, name='random_brightness')(noised_image)
contrasted_image = tfkl.RandomContrast(0.4, name='random_constrast')(brightened_image)
patches = ExtractPatchesLayer(patch_size)(contrasted_image)

mil_input = MILInputLayer(name="mil_input")(patches)
preprocessed_image = tf.keras.applications.resnet50.preprocess_input(mil_input)
backbone = tfk.applications.resnet50.ResNet50(include_top=False, weights='imagenet', input_tensor=preprocessed_image, pooling=None)

featureMaps = FeaturesMapsLayer(name="localFeatureMaps")(backbone.output)
gaps = tfkl.GlobalAveragePooling2D(name="localGaps")(featureMaps)

attentionWeights = GAMLayer(L, name="attentionWeights")(gaps)
reshaped_attentionWeights = ReshapeWeightsLayer(name="reshaped_w")([attentionWeights,input])
softmax_reshaped_attentionWeights = tf.keras.activations.softmax(reshaped_attentionWeights)
MIL_features = AggregateFeaturesLayer(name="MIL_features")([gaps,softmax_reshaped_attentionWeights])

dense = tfkl.Dense(128, activation="relu", kernel_regularizer=tfk.regularizers.l2(l2), name='local_dense1')(MIL_features)
dense = tfkl.Dense(64, activation="relu", kernel_regularizer=tfk.regularizers.l2(l2), name='local_dense2')(dense)
dense = tfkl.Dense(32, activation="relu", kernel_regularizer=tfk.regularizers.l2(l2), name='local_dense3')(dense)
output = tfkl.Dense(1, activation="sigmoid", kernel_regularizer=tfk.regularizers.l2(l2), name='local_output')(dense)

binary_classifier = tfk.Model(input, output, name="binary_classifier")

binary_classifier.summary()

# %%
# Create a learning rate scheduler callback.
lr_cb = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_binary_accuracy", mode="max", factor=0.25, patience=5)
# Create an early stopping callback.
es_cb = tf.keras.callbacks.EarlyStopping(monitor="val_binary_accuracy", mode="max", patience=8, restore_best_weights=True)

# Compile the model
binary_classifier.compile(
    # Optimizer
    optimizer=tfk.optimizers.Adam(learning_rate=learning_rate), 
    # Loss function to minimize
    loss='binary_crossentropy',
    # List of metrics to monitor
    metrics=[tfk.metrics.BinaryAccuracy(),tfk.metrics.Precision(),tfk.metrics.Recall()]
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


# %%
threshold = 0.5
true_labels = []
predicted_labels = []
for images,labels in test_ds:
    probabilities = binary_classifier.predict(images, verbose=0)
    predicted_labels += list((probabilities > threshold).astype(int))
    true_labels += list(labels.numpy())

accuracy = accuracy_score(true_labels, predicted_labels)
print(f'Accuracy: {accuracy:.4f}')

precision = precision_score(true_labels, predicted_labels)
print(f'Precision: {precision:.4f}')

recall = recall_score(true_labels, predicted_labels)
print(f'Recall: {recall:.4f}')

f1 = f1_score(true_labels, predicted_labels)
print(f'F1 Score: {f1:.4f}')

conf_matrix = confusion_matrix(true_labels, predicted_labels)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['0', '1'], yticklabels=['0', '1'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
