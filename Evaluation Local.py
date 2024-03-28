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
l2 = 1.0e-2
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
    
class GlobalHeatMapLayer(tf.keras.layers.Layer):
    def __init__(self, max_dim, **kwargs):
        super(GlobalHeatMapLayer, self).__init__(**kwargs)
        self.max_dim = max_dim

    def call(self, inputs, training=None):

        inputs[0] = tf.image.resize(inputs[0], size=(self.max_dim,self.max_dim))
        inputs[1] = tf.image.resize(inputs[1], size=(self.max_dim,self.max_dim))
        inputs[2] = tf.image.resize(inputs[2], size=(self.max_dim,self.max_dim))
        inputs[3] = tf.image.resize(inputs[3], size=(self.max_dim,self.max_dim))

        globalHeatmap = tfkl.Add()([inputs[0],inputs[1],inputs[2],inputs[3]])
        return globalHeatmap

    def get_config(self):
        config = super().get_config()
        config.update({'max_dim': self.max_dim})
        return config

class ExtractRelevantPatchesLayer(tf.keras.layers.Layer):
    def __init__(self, patch_size, num_patches, **kwargs):
        super(ExtractRelevantPatchesLayer, self).__init__(**kwargs)
        self.patch_size = patch_size
        self.num_patches = num_patches

    def call(self, inputs, training=None):
        heatmap_shape = tf.shape(inputs[0])
        image_shape = tf.shape(inputs[1])

        avg = tfkl.AveragePooling2D(
            pool_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            padding='valid'
        )(inputs[0])

        patches = tf.image.extract_patches(
            images=inputs[1],
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )

        avg_shape = tf.shape(avg)
        avg = tf.reshape(
            avg,
            shape=(avg_shape[0], avg_shape[1] * avg_shape[2] * avg_shape[3])
        )
        patches_shape = tf.shape(patches)
        patches = tf.reshape(
            patches,
            shape=(patches_shape[0], patches_shape[1] * patches_shape[2], patches_shape[3])
        )

        top_k_indices = tf.math.top_k(avg, self.num_patches).indices

        patches = tf.map_fn(lambda x: tf.gather(x[1], tf.math.top_k(x[0], self.num_patches).indices), (avg, patches), dtype=tf.float32)

        patches_shape = tf.shape(patches)
        patches = tf.reshape(
            patches,
            shape=(patches_shape[0] * patches_shape[1], self.patch_size, self.patch_size, 3)
        )

        return patches

    def get_config(self):
        config = super().get_config()
        config.update({'patch_size': self.patch_size})
        config.update({'num_patches': self.num_patches})
        return config
    
class LocalInputLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(LocalInputLayer, self).__init__(**kwargs)

    def call(self, inputs):

        return inputs

    def get_config(self):
        config = super(LocalInputLayer, self).get_config()
        return config

class LocalFeaturesMapsLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(LocalFeaturesMapsLayer, self).__init__(**kwargs)

    def call(self, inputs):

        return inputs

    def get_config(self):
        config = super(LocalFeaturesMapsLayer, self).get_config()
        return config

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
    
class AggregateLocalFeaturesLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AggregateLocalFeaturesLayer, self).__init__(**kwargs)

    def call(self, inputs):

        w_shape = tf.shape(inputs[1])
        gap_shape = tf.shape(inputs[0])

        w = tf.reshape(inputs[1], shape=(w_shape[0], w_shape[1], 1))
        gaps = tf.reshape(inputs[0], shape=(w_shape[0], w_shape[1], gap_shape[-1]))

        feature_vector = tf.multiply(gaps,w)
        feature_vector = tf.reduce_sum(feature_vector, axis=1)

        return feature_vector

    def get_config(self):
        config = super(AggregateLocalFeaturesLayer, self).get_config()
        return config
    
# %%
spatial_resolution = 20
max_dim = 1088
max_shape = tf.constant((max_dim, max_dim))
patch_size = 64

# Load classification model
binary_classifier = tfk.models.load_model(
    '/home/simones95/EnvCrime/Local20_PS64_K86.h5',
    custom_objects={'RandomNoiseLayer': RandomNoiseLayer,
                    'HideLayer':HideLayer,
                    'GlobalHeatMapLayer':GlobalHeatMapLayer,
                    'ExtractRelevantPatchesLayer':ExtractRelevantPatchesLayer,
                    'LocalInputLayer':LocalInputLayer,
                    'LocalFeaturesMapsLayer':LocalFeaturesMapsLayer,
                    'GAMLayer':GAMLayer,
                    'ReshapeWeightsLayer':ReshapeWeightsLayer,
                    'AggregateLocalFeaturesLayer':AggregateLocalFeaturesLayer})

patch_weight_model = tfk.Model(inputs=binary_classifier.get_layer('local_input').output,
                      outputs=[binary_classifier.get_layer('attentionWeights').output, binary_classifier.get_layer('localGaps').output] )

patch_probability_model = tfk.Model(inputs=binary_classifier.get_layer('aggregate_local_features_layer').output,
                      outputs=binary_classifier.get_layer('local_output').output)

# %%
shift = patch_size//8

threshold = [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,0.99]

sp_iou = np.zeros(len(threshold))
sp_dice = np.zeros(len(threshold))
sp_precision = np.zeros(len(threshold))
sp_recall = np.zeros(len(threshold))
sp_pixel_accuracy = np.zeros(len(threshold))

wp_iou = np.zeros(len(threshold))
wp_dice = np.zeros(len(threshold))
wp_precision = np.zeros(len(threshold))
wp_recall = np.zeros(len(threshold))
wp_pixel_accuracy = np.zeros(len(threshold))

swp_iou = np.zeros(len(threshold))
swp_dice = np.zeros(len(threshold))
swp_precision = np.zeros(len(threshold))
swp_recall = np.zeros(len(threshold))
swp_pixel_accuracy = np.zeros(len(threshold))

swp1_iou = np.zeros(len(threshold))
swp1_dice = np.zeros(len(threshold))
swp1_precision = np.zeros(len(threshold))
swp1_recall = np.zeros(len(threshold))
swp1_pixel_accuracy = np.zeros(len(threshold))

swp2_iou = np.zeros(len(threshold))
swp2_dice = np.zeros(len(threshold))
swp2_precision = np.zeros(len(threshold))
swp2_recall = np.zeros(len(threshold))
swp2_pixel_accuracy = np.zeros(len(threshold))

count = 0

for file_name in os.listdir('/dataset/new_simones/AW2_segmentation/'):
    count += 1
    print(count)

    # Load image
    image = tf.io.read_file("/dataset/new_simones/AerialWaste2/images/" + file_name)
    image = tf.image.decode_png(image, channels=3)

    segmentation = tf.io.read_file('/dataset/new_simones/AW2_segmentation/' + file_name)
    segmentation = tf.image.decode_png(segmentation, channels=1).numpy()

    segmentation[segmentation == 255] = 1

    # Spatial Resolution
    shape = tf.shape(image)
    spatial_resolution_image = 210 * 100 / tf.maximum(shape[0], shape[1])
    new_shape_image = tf.cast(tf.cast(shape, tf.float64) * (spatial_resolution_image / spatial_resolution), tf.int32)
    image = tf.image.resize(image, size=new_shape_image[0:2])
    segmentation = tf.image.resize(segmentation, size=new_shape_image[0:2])

    # Padding
    offset = max_shape - new_shape_image[0:2]
    image = tf.image.pad_to_bounding_box(image, offset[0] // 2, offset[1] // 2, max_dim, max_dim)
    segmentation = tf.image.pad_to_bounding_box(segmentation, offset[0] // 2, offset[1] // 2, max_dim, max_dim)

    segmentation = np.where(segmentation < 0.5, 0, 1)

    patches = tf.image.extract_patches(
        images=np.expand_dims(image,axis=0),
        sizes=[1, patch_size, patch_size, 1],
        strides=[1, shift, shift, 1],
        rates=[1, 1, 1, 1],
        padding='VALID'
    )

    patches_shape = tf.shape(patches)
    patches = tf.reshape(
        patches,
        shape=(patches_shape[0] * patches_shape[1]*patches_shape[2], patch_size, patch_size, 3)
    )

    weights, gaps = patch_weight_model.predict(patches,verbose=0)

    norm_weights = np.exp(weights - np.max(weights)) / np.sum(np.exp(weights - np.max(weights)))
    feature_vector = tf.reduce_sum(tf.multiply(gaps,norm_weights), axis=0)
    feature_vector = np.expand_dims(feature_vector, axis=0)
    all_features = np.vstack((gaps, feature_vector))

    local_probability = patch_probability_model.predict(all_features,verbose=0)

    sp = np.zeros((max_dim, max_dim, 1))
    wp = np.zeros((max_dim, max_dim, 1))
    swp = np.zeros((max_dim, max_dim, 1))
    w = np.zeros((max_dim, max_dim, 1))
    count_patch = 0
    for i in range(0, max_dim-patch_size+1, shift):
        for j in range(0, max_dim-patch_size + 1, shift):
            sp[i:i + patch_size, j:j + patch_size, :] += np.ones((patch_size, patch_size, 1)) * local_probability[count_patch, 0]
            wp[i:i + patch_size, j:j + patch_size, :] += np.ones((patch_size, patch_size, 1)) * weights[count_patch, 0]
            w[i:i + patch_size, j:j + patch_size, :] += np.ones((patch_size, patch_size, 1))
            count_patch += 1
    sp /= w
    wp /= w

    wp = np.clip(wp,0,np.max(wp))
    wp /= np.max(wp)

    swp = sp * wp
    swp1 = (swp - np.min(swp)) / (np.max(swp) - np.min(swp))
    swp2 = swp / np.max(swp)

    """
    fig, axs = plt.subplots(1, 5, figsize=(20, 5))
    axs[0].imshow(sp, cmap='jet', alpha=0.8)
    axs[0].imshow(image/255, alpha=0.5)
    axs[0].axis('off')
    axs[1].imshow(wp, cmap='jet', alpha=0.8)
    axs[1].imshow(image/255, alpha=0.5)
    axs[1].axis('off')
    axs[2].imshow(swp, cmap='jet', alpha=0.8)
    axs[2].imshow(image/255, alpha=0.5)
    axs[2].axis('off')
    axs[3].imshow(swp1, cmap='jet', alpha=0.8)
    axs[3].imshow(image/255, alpha=0.5)
    axs[3].axis('off')
    axs[4].imshow(swp2, cmap='jet', alpha=0.8)
    axs[4].imshow(image/255, alpha=0.5)
    axs[4].axis('off')
    plt.show()
    """

    for index in range(len(threshold)):

      predicted_segmentation = np.where(sp < threshold[index], 0, 1)

      conf_matrix = confusion_matrix(segmentation.flatten(), predicted_segmentation.flatten())

      TP = conf_matrix[1, 1]  # True Positives
      FP = conf_matrix[0, 1]  # False Positives
      FN = conf_matrix[1, 0]  # False Negatives
      TN = conf_matrix[0, 0]  # True Negatives

      sp_iou[index] += TP / (TP + FP + FN)
      sp_dice[index] += (2 * TP) / (2 * TP + FP + FN)
      if (TP + FP) != 0:
        sp_precision[index] += TP / (TP + FP)
      if (TP + FN) != 0:
        sp_recall[index] += TP / (TP + FN)
      sp_pixel_accuracy[index] += (TP + TN) / np.sum(conf_matrix)

      predicted_segmentation = np.where(wp < threshold[index], 0, 1)

      conf_matrix = confusion_matrix(segmentation.flatten(), predicted_segmentation.flatten())

      TP = conf_matrix[1, 1]  # True Positives
      FP = conf_matrix[0, 1]  # False Positives
      FN = conf_matrix[1, 0]  # False Negatives
      TN = conf_matrix[0, 0]  # True Negatives

      wp_iou[index] += TP / (TP + FP + FN)
      wp_dice[index] += (2 * TP) / (2 * TP + FP + FN)
      if (TP + FP) != 0:
        wp_precision[index] += TP / (TP + FP)
      if (TP + FN) != 0:
        wp_recall[index] += TP / (TP + FN)
      wp_pixel_accuracy[index] += (TP + TN) / np.sum(conf_matrix)

      predicted_segmentation = np.where(swp < threshold[index], 0, 1)

      conf_matrix = confusion_matrix(segmentation.flatten(), predicted_segmentation.flatten())

      TP = conf_matrix[1, 1]  # True Positives
      FP = conf_matrix[0, 1]  # False Positives
      FN = conf_matrix[1, 0]  # False Negatives
      TN = conf_matrix[0, 0]  # True Negatives

      swp_iou[index] += TP / (TP + FP + FN)
      swp_dice[index] += (2 * TP) / (2 * TP + FP + FN)
      if (TP + FP) != 0:
        swp_precision[index] += TP / (TP + FP)
      if (TP + FN) != 0:
        swp_recall[index] += TP / (TP + FN)
      swp_pixel_accuracy[index] += (TP + TN) / np.sum(conf_matrix)

      predicted_segmentation = np.where(swp1 < threshold[index], 0, 1)

      conf_matrix = confusion_matrix(segmentation.flatten(), predicted_segmentation.flatten())

      TP = conf_matrix[1, 1]  # True Positives
      FP = conf_matrix[0, 1]  # False Positives
      FN = conf_matrix[1, 0]  # False Negatives
      TN = conf_matrix[0, 0]  # True Negatives

      swp1_iou[index] += TP / (TP + FP + FN)
      swp1_dice[index] += (2 * TP) / (2 * TP + FP + FN)
      if (TP + FP) != 0:
        swp1_precision[index] += TP / (TP + FP)
      if (TP + FN) != 0:
        swp1_recall[index] += TP / (TP + FN)
      swp1_pixel_accuracy[index] += (TP + TN) / np.sum(conf_matrix)

      predicted_segmentation = np.where(swp2 < threshold[index], 0, 1)

      conf_matrix = confusion_matrix(segmentation.flatten(), predicted_segmentation.flatten())

      TP = conf_matrix[1, 1]  # True Positives
      FP = conf_matrix[0, 1]  # False Positives
      FN = conf_matrix[1, 0]  # False Negatives
      TN = conf_matrix[0, 0]  # True Negatives

      swp2_iou[index] += TP / (TP + FP + FN)
      swp2_dice[index] += (2 * TP) / (2 * TP + FP + FN)
      if (TP + FP) != 0:
        swp2_precision[index] += TP / (TP + FP)
      if (TP + FN) != 0:
        swp2_recall[index] += TP / (TP + FN)
      swp2_pixel_accuracy[index] += (TP + TN) / np.sum(conf_matrix)

sp_iou /= count
sp_dice /= count
sp_precision /= count
sp_recall /= count
sp_pixel_accuracy /= count

wp_iou /= count
wp_dice /= count
wp_precision /= count
wp_recall /= count
wp_pixel_accuracy /= count

swp_iou /= count
swp_dice /= count
swp_precision /= count
swp_recall /= count
swp_pixel_accuracy /= count

swp1_iou /= count
swp1_dice /= count
swp1_precision /= count
swp1_recall /= count
swp1_pixel_accuracy /= count

swp2_iou /= count
swp2_dice /= count
swp2_precision /= count
swp2_recall /= count
swp2_pixel_accuracy /= count

index_max_sp = np.argmax(sp_iou)
index_max_wp = np.argmax(wp_iou)
index_max_swp = np.argmax(swp_iou)

index_max_swp1 = np.argmax(swp1_iou)
index_max_swp2 = np.argmax(swp2_iou)

print("SP")
print("Threshold: ", threshold[index_max_sp])
print("IoU: ", sp_iou[index_max_sp])
print("Dice: ",sp_dice[index_max_sp])
print("Precision: ",sp_precision[index_max_sp])
print("Recall: ",sp_recall[index_max_sp])
f1 = (2 * sp_precision[index_max_sp] * sp_recall[index_max_sp])/(sp_precision[index_max_sp] + sp_recall[index_max_sp])
print("F1-Score: ",f1)
print("Pixel-Accuracy: ",sp_pixel_accuracy[index_max_sp])
print("#####################################")

print("WP")
print("Threshold: ", threshold[index_max_wp])
print("IoU: ", wp_iou[index_max_wp])
print("Dice: ",wp_dice[index_max_wp])
print("Precision: ",wp_precision[index_max_wp])
print("Recall: ",wp_recall[index_max_wp])
f1 = (2 * wp_precision[index_max_wp] * wp_recall[index_max_wp])/(wp_precision[index_max_wp] + wp_recall[index_max_wp])
print("F1-Score: ",f1)
print("Pixel-Accuracy: ",wp_pixel_accuracy[index_max_wp])
print("#####################################")

print("SWP")
print("Threshold: ", threshold[index_max_swp])
print("IoU: ", swp_iou[index_max_swp])
print("Dice: ",swp_dice[index_max_swp])
print("Precision: ",swp_precision[index_max_swp])
print("Recall: ",swp_recall[index_max_swp])
f1 = (2 * swp_precision[index_max_swp] * swp_recall[index_max_swp])/(swp_precision[index_max_swp] + swp_recall[index_max_swp])
print("F1-Score: ",f1)
print("Pixel-Accuracy: ",swp_pixel_accuracy[index_max_swp])
print("#####################################")

print("swp1")
print("Threshold: ", threshold[index_max_swp1])
print("IoU: ", swp1_iou[index_max_swp1])
print("Dice: ",swp1_dice[index_max_swp1])
print("Precision: ",swp1_precision[index_max_swp1])
print("Recall: ",swp1_recall[index_max_swp1])
f1 = (2 * swp1_precision[index_max_swp1] * swp1_recall[index_max_swp1])/(swp1_precision[index_max_swp1] + swp1_recall[index_max_swp1])
print("F1-Score: ",f1)
print("Pixel-Accuracy: ",swp1_pixel_accuracy[index_max_swp1])
print("#####################################")


print("swp2")
print("Threshold: ", threshold[index_max_swp2])
print("IoU: ", swp2_iou[index_max_swp2])
print("Dice: ",swp2_dice[index_max_swp2])
print("Precision: ",swp2_precision[index_max_swp2])
print("Recall: ",swp2_recall[index_max_swp2])
f1 = (2 * swp2_precision[index_max_swp2] * swp2_recall[index_max_swp2])/(swp2_precision[index_max_swp2] + swp2_recall[index_max_swp2])
print("F1-Score: ",f1)
print("Pixel-Accuracy: ",swp2_pixel_accuracy[index_max_swp2])
print("#####################################")

# %%
