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
import minerl


def Xception(img_input=None):
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

    x = layers.Conv2D(8, (3, 3), strides=(2, 2), use_bias=False, name='block1_conv1')(img_input)
    x = layers.BatchNormalization(name='block1_conv1_bn')(x)
    x = layers.Activation('relu', name='block1_conv1_act')(x)

    x = layers.Conv2D(16, (3, 3), use_bias=False, name='block1_conv2')(x)
    x = layers.BatchNormalization(name='block1_conv2_bn')(x)
    x = layers.Activation('relu', name='block1_conv2_act')(x)

    residual = layers.Conv2D(32, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = layers.BatchNormalization()(residual)

    x = layers.SeparableConv2D(32, (3, 3), padding='same', use_bias=False, name='block2_sepconv1')(x)
    x = layers.BatchNormalization(name='block2_sepconv1_bn')(x)
    x = layers.Activation('relu', name='block2_sepconv2_act')(x)

    x = layers.SeparableConv2D(32, (3, 3), padding='same', use_bias=False, name='block2_sepconv2')(x)
    x = layers.BatchNormalization(name='block2_sepconv2_bn')(x)

    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block2_pool')(x)
    x = layers.add([x, residual])

    residual = layers.Conv2D(64, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = layers.BatchNormalization()(residual)

    x = layers.Activation('relu', name='block3_sepconv1_act')(x)

    x = layers.SeparableConv2D(64, (3, 3), padding='same', use_bias=False, name='block3_sepconv1')(x)
    x = layers.BatchNormalization(name='block3_sepconv1_bn')(x)
    x = layers.Activation('relu', name='block3_sepconv2_act')(x)

    x = layers.SeparableConv2D(64, (3, 3), padding='same', use_bias=False, name='block3_sepconv2')(x)
    x = layers.BatchNormalization(name='block3_sepconv2_bn')(x)

    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block3_pool')(x)
    x = layers.add([x, residual])

    residual = layers.Conv2D(128, (1, 1),
                             strides=(2, 2),
                             padding='same',
                             use_bias=False)(x)
    residual = layers.BatchNormalization()(residual)

    x = layers.Activation('relu', name='block4_sepconv1_act')(x)

    x = layers.SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='block4_sepconv1')(x)
    x = layers.BatchNormalization(name='block4_sepconv1_bn')(x)
    x = layers.Activation('relu', name='block4_sepconv2_act')(x)

    x = layers.SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='block4_sepconv2')(x)
    x = layers.BatchNormalization(name='block4_sepconv2_bn')(x)

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
    #     x = layers.BatchNormalization(
    #                                   name=prefix + '_sepconv1_bn')(x)
    #     x = layers.Activation('relu', name=prefix + '_sepconv2_act')(x)
    #     x = layers.SeparableConv2D(728, (3, 3),
    #                                padding='same',
    #                                use_bias=False,
    #                                name=prefix + '_sepconv2')(x)
    #     x = layers.BatchNormalization(
    #                                   name=prefix + '_sepconv2_bn')(x)
    #     x = layers.Activation('relu', name=prefix + '_sepconv3_act')(x)
    #     x = layers.SeparableConv2D(728, (3, 3),
    #                                padding='same',
    #                                use_bias=False,
    #                                name=prefix + '_sepconv3')(x)
    #     x = layers.BatchNormalization(
    #                                   name=prefix + '_sepconv3_bn')(x)
    #
    #     x = layers.add([x, residual])

    residual = layers.Conv2D(256, (1, 1), strides=(2, 2),
                             padding='same', use_bias=False)(x)
    residual = layers.BatchNormalization()(residual)

    x = layers.Activation('relu', name='block13_sepconv1_act')(x)
    x = layers.SeparableConv2D(256, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block13_sepconv1')(x)
    x = layers.BatchNormalization(name='block13_sepconv1_bn')(x)
    x = layers.Activation('relu', name='block13_sepconv2_act')(x)
    x = layers.SeparableConv2D(256, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block13_sepconv2')(x)
    x = layers.BatchNormalization(name='block13_sepconv2_bn')(x)

    x = layers.MaxPooling2D((3, 3),
                            strides=(2, 2),
                            padding='same',
                            name='block13_pool')(x)
    x = layers.add([x, residual])

    x = layers.SeparableConv2D(256, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block14_sepconv1')(x)
    x = layers.BatchNormalization(name='block14_sepconv1_bn')(x)
    x = layers.Activation('relu', name='block14_sepconv1_act')(x)

    x = layers.SeparableConv2D(256, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block14_sepconv2')(x)
    x = layers.BatchNormalization(name='block14_sepconv2_bn')(x)
    x = layers.Activation('relu', name='block14_sepconv2_act')(x)

    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = layers.Dense(15, activation='sigmoid', name='xception_output')(x)

    # Create model.
    model = models.Model(img_input, x, name='xception')
    # model.compile(optimizer='adam', loss='mean_squared_error')
    # if weights_path != None: model.load_weights(weights_path)

    return model


def fancy_nn(weights_path=None):
    """Instantiates the Xception architecture.

        Optionally loads weights pre-trained on ImageNet.
        Note that the data format convention used by the model is
        the one specified in your Keras config at `~/.keras/keras.json`.

        Note that the default input image size for this model is 299x299.

        # Arguments
            weights_path = None:
                Path from where weights are to be loaded into the model after compilation.

        # Returns
            A Keras model instance.

        ## Raises
        #    ValueError: in case of invalid argument for `weights`,
        #        or invalid input shape.
        #    RuntimeError: If attempting to run this model with a
        #        backend that does not support separable convolutions.
        """

    vanilla_input = layers.Input(shape=(21,))
    img_input = layers.Input(shape=(64, 64, 3))

    vanilla = layers.Dense(15, name='hidden')(vanilla_input)
    vanilla = models.Model(inputs=vanilla_input, outputs=vanilla, name='vanilla_hidden')

    x = layers.concatenate([vanilla.output, Xception(img_input).output], name='hidden_concatenated')

    binary_output = layers.Dense(13, name='binary_prediction')(x)
    linear_output = layers.Dense(2, activation='linear', name='linear_prediction')(x)

    binary_model = models.Model(inputs=[vanilla_input, img_input], outputs=binary_output)
    linear_model = models.Model(inputs=[vanilla_input, img_input], outputs=linear_output)

    model = models.Model(inputs=[vanilla_input, img_input], outputs=[binary_model.output, linear_model.output])

    model.compile(optimizer='adam', loss='mean_squared_error')
    if weights_path != None: model.load_weights(weights_path)

    return model


def state_to_inputs(state):
    linear_inputs = [
        int(state['equipped_items']['mainhand']['damage']),
        int(state['equipped_items']['mainhand']['maxDamage']),
        int(state['equipped_items']['mainhand']['type']),
        int(state['inventory']['coal']),
        int(state['inventory']['cobblestone']),
        int(state['inventory']['crafting_table']),
        int(state['inventory']['dirt']),
        int(state['inventory']['furnace']),
        int(state['inventory']['iron_axe']),
        int(state['inventory']['iron_ingot']),
        int(state['inventory']['iron_ore']),
        int(state['inventory']['iron_pickaxe']),
        int(state['inventory']['log']),
        int(state['inventory']['planks']),
        int(state['inventory']['stick']),
        int(state['inventory']['stone']),
        int(state['inventory']['stone_axe']),
        int(state['inventory']['stone_pickaxe']),
        int(state['inventory']['torch']),
        int(state['inventory']['wooden_axe']),
        int(state['inventory']['wooden_pickaxe'])
    ]

    img_input = state['pov']

    return [linear_inputs, img_input]


def label_to_output(labels):
    binary_labels = []
    binary_labels.append(labels['attack'])
    binary_labels.append(labels['back'])
    binary_labels.append(labels['forward'])
    binary_labels.append(labels['jump'])
    binary_labels.append(labels['left'])
    binary_labels.append(labels['right'])
    binary_labels.append(labels['sneak'])
    binary_labels.append(labels['sprint'])

    linear_labels = []
    linear_labels.append(int(labels['camera'][0]))
    linear_labels.append(int(labels['camera'][1]))
    linear_labels.append(labels['craft'])
    linear_labels.append(labels['equip'])
    linear_labels.append(labels['nearbyCraft'])
    linear_labels.append(labels['nearbySmelt'])
    linear_labels.append(labels['place'])

    return [binary_labels, linear_labels]


def outputs_to_action(outputs):
    binary_labels = outputs[0]
    linear_labels = outputs[1]

    from collections import OrderedDict
    from numpy import array, float32

    action = OrderedDict({
        'attack': binary_labels[0],
        'back': binary_labels[1],
        'camera': array([linear_labels[0][0], linear_labels[0][1]], dtype=float32),
        'craft': linear_labels[1],
        'equip': linear_labels[2],
        'forward': binary_labels[2],
        'jump': binary_labels[3],
        'left': binary_labels[4],
        'nearbyCraft': linear_labels[3],
        'nearbySmelt': linear_labels[4],
        'place': linear_labels[5],
        'right': binary_labels[5],
        'sneak': binary_labels[6],
        'sprint': binary_labels[7]
    })

    return action


if __name__ == '__main__':
    model = fancy_nn()

    # from keras.utils import plot_model
    # plot_model(model, to_file='model.png')

    inputs = state_to_inputs(minerl.env.obtain_observation_space.sample())
    labels = label_to_output(minerl.env.obtain_action_space.sample())

    # print(len(inputs))
    # print(inputs)
    # print(len(inputs[0]))
    # print('\n')
    # print(inputs[0])
    # print('\n')
    # print(len(inputs[1]))
    # print('\n')
    # print(inputs[1])
    # print('\n')

    predictions = model.predict(inputs)
    # action = outputs_to_action(predictions)
    # print(action)

    # print(minerl.env.obtain_action_space.sample())
    # Xception(minerl.env.obtain_observation_space, minerl.env.obtain_action_space.sample())
    for i in range(10):
        pass
        # print(minerl.env.obtain_action_space.sample())
        """[
            ('attack', 1),
            ('back', 1),
            ('camera', array([134.4457 , 173.21553], dtype=float32)),
            ('craft', 1),
            ('equip', 5),
            ('forward', 1),
            ('jump', 0),
            ('left', 0),
            ('nearbyCraft', 3),
            ('nearbySmelt', 0),
            ('place', 1),
            ('right', 0),
            ('sneak', 1),
            ('sprint', 0)
        ]"""
        # [('attack', 1), ('back', 1), ('camera', array([-11.986441, -67.31623 ], dtype=float32)), ('craft', 0), ('equip', 0), ('forward', 1), ('jump', 0), ('left', 1), ('nearbyCraft', 5), ('nearbySmelt', 2), ('place', 3), ('right', 1), ('sneak', 0), ('sprint', 0)]
        # [('attack', 1), ('back', 1), ('camera', array([ 114.31422, -140.80772], dtype=float32)), ('craft', 4), ('equip', 5), ('forward', 0), ('jump', 1), ('left', 0), ('nearbyCraft', 7), ('nearbySmelt', 1), ('place', 2), ('right', 1), ('sneak', 1), ('sprint', 1)]
        # [('attack', 0), ('back', 0), ('camera', array([  71.87382, -169.42906], dtype=float32)), ('craft', 4), ('equip', 3), ('forward', 1), ('jump', 0), ('left', 1), ('nearbyCraft', 7), ('nearbySmelt', 0), ('place', 2), ('right', 1), ('sneak', 1), ('sprint', 0)]
        # [('attack', 0), ('back', 0), ('camera', array([  -0.22040889, -108.96448   ], dtype=float32)), ('craft', 1), ('equip', 4), ('forward', 0), ('jump', 1), ('left', 1), ('nearbyCraft', 6), ('nearbySmelt', 0), ('place', 2), ('right', 0), ('sneak', 1), ('sprint', 1)]
        # [('attack', 1), ('back', 0), ('camera', array([ 42.048176, 114.77076 ], dtype=float32)), ('craft', 0), ('equip', 0), ('forward', 1), ('jump', 1), ('left', 0), ('nearbyCraft', 2), ('nearbySmelt', 1), ('place', 5), ('right', 0), ('sneak', 0), ('sprint', 1)]
        # [('attack', 0), ('back', 1), ('camera', array([  28.670277, -154.99754 ], dtype=float32)), ('craft', 4), ('equip', 2), ('forward', 0), ('jump', 0), ('left', 0), ('nearbyCraft', 7), ('nearbySmelt', 2), ('place', 6), ('right', 0), ('sneak', 0), ('sprint', 0)]
        # [('attack', 0), ('back', 1), ('camera', array([-137.78952,  159.17415], dtype=float32)), ('craft', 1), ('equip', 1), ('forward', 0), ('jump', 1), ('left', 1), ('nearbyCraft', 0), ('nearbySmelt', 1), ('place', 5), ('right', 1), ('sneak', 0), ('sprint', 1)]
        # [('attack', 0), ('back', 0), ('camera', array([ 38.89587, 148.0062 ], dtype=float32)), ('craft', 0), ('equip', 6), ('forward', 1), ('jump', 1), ('left', 1), ('nearbyCraft', 2), ('nearbySmelt', 1), ('place', 0), ('right', 0), ('sneak', 0), ('sprint', 1)]
        # [('attack', 1), ('back', 0), ('camera', array([158.30092 ,  84.317894], dtype=float32)), ('craft', 4), ('equip', 3), ('forward', 1), ('jump', 0), ('left', 0), ('nearbyCraft', 4), ('nearbySmelt', 2), ('place', 1), ('right', 1), ('sneak', 0), ('sprint', 0)]
        # [('attack', 0), ('back', 0), ('camera', array([-135.29819,  -26.28303], dtype=float32)), ('craft', 3), ('equip', 0), ('forward', 1), ('jump', 1), ('left', 1), ('nearbyCraft', 3), ('nearbySmelt', 2), ('place', 5), ('right', 0), ('sneak', 1), ('sprint', 0)]

        # print(minerl.env.obtain_observation_space.sample())
        """[
            ('equipped_items', OrderedDict(
                [('mainhand', OrderedDict(
                        [
                            ('damage', array(291)),
                            ('maxDamage', array(136)),
                            ('type', 7)
                        ]
                ))]
            )),
            ('inventory', OrderedDict(
                [
                    ('coal', array(1777)),
                    ('cobblestone', array(2185)),
                    ('crafting_table', array(926)),
                    ('dirt', array(2150)),
                    ('furnace', array(1135)),
                    ('iron_axe', array(547)),
                    ('iron_ingot', array(446)),
                    ('iron_ore', array(1711)),
                    ('iron_pickaxe', array(984)),
                    ('log', array(1037)),
                    ('planks', array(1337)),
                    ('stick', array(1221)),
                    ('stone', array(713)),
                    ('stone_axe', array(315)),
                    ('stone_pickaxe', array(2070)),
                    ('torch', array(364)),
                    ('wooden_axe', array(1056)),
                    ('wooden_pickaxe', array(2145))
                ]
            ))
        ]"""
        # 21 items
