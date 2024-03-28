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
spatial_resolution = 30 #20
max_dim = 736 #1088
max_shape = tf.constant((max_dim, max_dim))

crop_shape = 256
translation_rate = 0.1

learning_rate = 1.0e-5
l2 = 1.0e-2

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
  plt.figure(figsize=(20, 20))
  for index in range(2):
    print(o[index])
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
# Model

input = tfkl.Input(shape=(max_dim, max_dim, 3), name='input')
translated_image = tfkl.RandomTranslation(height_factor=translation_rate, width_factor=translation_rate, fill_mode='reflect', name="translated_image")(input)
central_crop = tfkl.CenterCrop(height=crop_shape, width=crop_shape, name="central_crop")(translated_image)
flipped_image = tfkl.RandomFlip(name='random_flip')(central_crop)
rotated_image = tfkl.RandomRotation(1, fill_mode='constant', fill_value=0.0, name='random_rotation')(flipped_image)
noised_image = RandomNoiseLayer(20.0, name='random_noise')(rotated_image)
brightened_image = tfkl.RandomBrightness(0.4, name='random_brightness')(noised_image)
contrasted_image = tfkl.RandomContrast(0.4, name='random_constrast')(brightened_image)

#preprocessed_image = tf.keras.applications.resnet50.preprocess_input(contrasted_image)
#backbone = tfk.applications.resnet50.ResNet50(include_top=False, weights='imagenet', input_tensor=preprocessed_image, pooling='avg')

#preprocessed_image = tf.keras.applications.inception_resnet_v2.preprocess_input(contrasted_image)
#backbone = tfk.applications.inception_resnet_v2.InceptionResNetV2 (include_top=False, weights='imagenet', input_tensor=preprocessed_image, pooling='avg')

preprocessed_image = tf.keras.applications.vgg19.preprocess_input(contrasted_image)
backbone = tfk.applications.vgg19.VGG19 (include_top=False, weights='imagenet', input_tensor=preprocessed_image, pooling='avg')


dense = tfkl.Dense(128, activation="relu", kernel_regularizer=tfk.regularizers.l2(l2), name='dense1')(backbone.output)
dense = tfkl.Dense(64, activation="relu", kernel_regularizer=tfk.regularizers.l2(l2), name='dense2')(dense)
dense = tfkl.Dense(32, activation="relu", kernel_regularizer=tfk.regularizers.l2(l2), name='dense3')(dense)
output = tfkl.Dense(1, activation="sigmoid", kernel_regularizer=tfk.regularizers.l2(l2), name='output')(dense)

binary_classifier = tfk.Model(input, output, name="binary_classifier")

# Print the summary of the model
binary_classifier.summary()

# %%
# Create a learning rate scheduler callback.
lr_cb = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_binary_accuracy", mode="max", factor=0.25, patience=5)
# Create an early stopping callback.
es_cb = tf.keras.callbacks.EarlyStopping(monitor="val_binary_accuracy", mode="max", patience=8, restore_best_weights=True)

# Compile the model
binary_classifier.compile(
    optimizer=tfk.optimizers.Adam(learning_rate=learning_rate),  # Optimizer
    # Loss function to minimize
    loss=tfk.losses.BinaryCrossentropy(),
    # List of metrics to monitor
    metrics=[tfk.metrics.BinaryAccuracy(), tfk.metrics.Precision(), tfk.metrics.Recall()]
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
