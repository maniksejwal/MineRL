"""Xception V1 model for Keras.

On ImageNet, this model gets to a top-1 validation accuracy of 0.790
and a top-5 validation accuracy of 0.945.

Do note that the input image format for this model is different than for
the VGG16 and ResNet models (299x299 instead of 224x224),
and that the input preprocessing function
is also different (same as Inception V3).

# Reference

- [Xception: Deep Learning with Depthwise Separable Convolutions](
    https://arxiv.org/abs/1610.02357) (CVPR 2017)

"""

from keras import layers, models
import tensorflow

def Xception(actions=None, weights_path=None):
    """Instantiates the Xception architecture.

    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.

    Note that the default input image size for this model is 299x299.

    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor
            (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(299, 299, 3)`.
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 71.
            E.g. `(150, 150, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional block.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional block, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True,
            and if no `weights` argument is specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
        RuntimeError: If attempting to run this model with a
            backend that does not support separable convolutions.
    """

    img_input = layers.Input(shape=(64,64,3))
    channel_axis = 1

    x = layers.Conv2D(8, (3, 3), strides=(2, 2), use_bias=False, name='block1_conv1')(img_input)
    x = layers.BatchNormalization(axis=channel_axis, name='block1_conv1_bn')(x)
    x = layers.Activation('relu', name='block1_conv1_act')(x)

    x = layers.Conv2D(16, (3, 3), use_bias=False, name='block1_conv2')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block1_conv2_bn')(x)
    x = layers.Activation('relu', name='block1_conv2_act')(x)

    residual = layers.Conv2D(32, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = layers.BatchNormalization(axis=channel_axis)(residual)

    x = layers.SeparableConv2D(32, (3, 3), padding='same', use_bias=False, name='block2_sepconv1')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block2_sepconv1_bn')(x)
    x = layers.Activation('relu', name='block2_sepconv2_act')(x)

    x = layers.SeparableConv2D(32, (3, 3), padding='same', use_bias=False, name='block2_sepconv2')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block2_sepconv2_bn')(x)

    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block2_pool')(x)
    x = layers.add([x, residual])

    residual = layers.Conv2D(64, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = layers.BatchNormalization(axis=channel_axis)(residual)

    x = layers.Activation('relu', name='block3_sepconv1_act')(x)

    x = layers.SeparableConv2D(64, (3, 3), padding='same', use_bias=False, name='block3_sepconv1')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block3_sepconv1_bn')(x)
    x = layers.Activation('relu', name='block3_sepconv2_act')(x)

    x = layers.SeparableConv2D(64, (3, 3), padding='same', use_bias=False, name='block3_sepconv2')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block3_sepconv2_bn')(x)

    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block3_pool')(x)
    x = layers.add([x, residual])

    residual = layers.Conv2D(128, (1, 1),
                             strides=(2, 2),
                             padding='same',
                             use_bias=False)(x)
    residual = layers.BatchNormalization(axis=channel_axis)(residual)

    x = layers.Activation('relu', name='block4_sepconv1_act')(x)

    x = layers.SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='block4_sepconv1')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block4_sepconv1_bn')(x)
    x = layers.Activation('relu', name='block4_sepconv2_act')(x)

    x = layers.SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='block4_sepconv2')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block4_sepconv2_bn')(x)

    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block4_pool')(x)
    x = layers.add([x, residual])

    # for i in range(8):
    #     residual = x
    #     prefix = 'block' + str(i + 5)
    #
    #     x = layers.Activation('relu', name=prefix + '_sepconv1_act')(x)
    #     x = layers.SeparableConv2D(728, (3, 3),
    #                                padding='same',
    #                                use_bias=False,
    #                                name=prefix + '_sepconv1')(x)
    #     x = layers.BatchNormalization(axis=channel_axis,
    #                                   name=prefix + '_sepconv1_bn')(x)
    #     x = layers.Activation('relu', name=prefix + '_sepconv2_act')(x)
    #     x = layers.SeparableConv2D(728, (3, 3),
    #                                padding='same',
    #                                use_bias=False,
    #                                name=prefix + '_sepconv2')(x)
    #     x = layers.BatchNormalization(axis=channel_axis,
    #                                   name=prefix + '_sepconv2_bn')(x)
    #     x = layers.Activation('relu', name=prefix + '_sepconv3_act')(x)
    #     x = layers.SeparableConv2D(728, (3, 3),
    #                                padding='same',
    #                                use_bias=False,
    #                                name=prefix + '_sepconv3')(x)
    #     x = layers.BatchNormalization(axis=channel_axis,
    #                                   name=prefix + '_sepconv3_bn')(x)
    #
    #     x = layers.add([x, residual])

    residual = layers.Conv2D(256, (1, 1), strides=(2, 2),
                             padding='same', use_bias=False)(x)
    residual = layers.BatchNormalization(axis=channel_axis)(residual)

    x = layers.Activation('relu', name='block13_sepconv1_act')(x)
    x = layers.SeparableConv2D(256, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block13_sepconv1')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block13_sepconv1_bn')(x)
    x = layers.Activation('relu', name='block13_sepconv2_act')(x)
    x = layers.SeparableConv2D(256, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block13_sepconv2')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block13_sepconv2_bn')(x)

    x = layers.MaxPooling2D((3, 3),
                            strides=(2, 2),
                            padding='same',
                            name='block13_pool')(x)
    x = layers.add([x, residual])

    x = layers.SeparableConv2D(256, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block14_sepconv1')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block14_sepconv1_bn')(x)
    x = layers.Activation('relu', name='block14_sepconv1_act')(x)

    x = layers.SeparableConv2D(256, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block14_sepconv2')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block14_sepconv2_bn')(x)
    x = layers.Activation('relu', name='block14_sepconv2_act')(x)

    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = layers.Dense(actions, activation='softmax', name='predictions')(x)

    inputs = img_input

    # Create model.
    model = models.Model(inputs, x, name='xception')

    if weights_path != None: model.load_weights(weights_path)

    return model

if __name__ == '__main__':
    Xception([True, False])