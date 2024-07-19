import tensorflow as tf
from tensorflow.keras import layers, Model

# this model is a mobilenetV2-based implementation of pix2pix architecture, for a 400x240 input
# mobilenetV2 is pretrained with images from imagenet

# encode feature layers
def MobileNetV2_encoder(input_shape):
    base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    base_model.trainable = False  # Freeze the base model

    # extract feature layers
    layer_names = [
        'block_1_expand_relu',   # 200x120
        'block_3_expand_relu',   # 100x60
        'block_6_expand_relu',   # 50x30
        'block_13_expand_relu',  # 25x15
        'block_16_project',      # 13x8
    ]
    layers = [base_model.get_layer(name).output for name in layer_names]
    
    encoder = Model(inputs=base_model.input, outputs=layers)
    return encoder

input_shape = (400, 240, 3)
encoder = MobileNetV2_encoder(input_shape)



# upsampler is used by decoder
def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    
    result = tf.keras.Sequential()
    result.add(layers.Conv2DTranspose(filters, size, strides=2, padding='same', 
                                      kernel_initializer=initializer, use_bias=False))

    result.add(layers.BatchNormalization())

    if apply_dropout:
        result.add(layers.Dropout(0.5))

    result.add(layers.ReLU())
    
    return result



# the decoder upsamples feature maps back to original image size
def Pix2Pix_decoder(input_shape=(400, 240, 3)):
    inputs = layers.Input(shape=input_shape)

    down_stack = MobileNetV2_encoder(input_shape)

    # Adjusting upsampling layers to match the new input dimensions
    up_stack = [
        upsample(512, 4, apply_dropout=True),  # 13x8 -> 26x16
        upsample(256, 4, apply_dropout=True),  # 26x16 -> 52x32
        upsample(128, 4, apply_dropout=True),  # 52x32 -> 104x64
        upsample(64, 4),                       # 104x64 -> 208x128
        upsample(32, 4),                       # 208x128 -> 416x256
        # Since the input size is not perfectly doubling at each step, we need to add an additional layer to resize to the exact target shape
        layers.Conv2DTranspose(16, (4, 4), strides=(2, 2), padding='same', kernel_initializer=tf.random_normal_initializer(0., 0.02), use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Conv2DTranspose(3, (4, 4), strides=(1, 1), padding='same', kernel_initializer=tf.random_normal_initializer(0., 0.02), activation='tanh')
    ]

    x = inputs

    # Downsampling through the model
    skips = down_stack(x)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack[:-1], skips):
        x = up(x)
        x = layers.Concatenate()([x, skip])

    x = up_stack[-1](x)  # Final upsampling layer to match the exact target shape

    return Model(inputs=inputs, outputs=x)

input_shape = (400, 240, 3)
generator = Pix2Pix_decoder(input_shape)



# define the generator, descriminator, and Pix2Pix model
def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        layers.Conv2D(filters, size, strides=2, padding='same',
                      kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(layers.BatchNormalization())

    result.add(layers.LeakyReLU())

    return result

def Discriminator(input_shape=(400, 240, 3)):
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = layers.Input(shape=input_shape, name='input_image')
    tar = layers.Input(shape=input_shape, name='target_image')

    x = layers.concatenate([inp, tar])  # (bs, 400, 240, channels*2)

    down1 = downsample(64, 4, False)(x)  # (bs, 200, 120, 64)
    down2 = downsample(128, 4)(down1)  # (bs, 100, 60, 128)
    down3 = downsample(256, 4)(down2)  # (bs, 50, 30, 256)
    down4 = downsample(512, 4)(down3)  # (bs, 25, 15, 512)

    zero_pad1 = layers.ZeroPadding2D()(down4)  # (bs, 27, 17, 512)
    conv = layers.Conv2D(512, 4, strides=1,
                         kernel_initializer=initializer,
                         use_bias=False)(zero_pad1)  # (bs, 24, 14, 512)

    batchnorm1 = layers.BatchNormalization()(conv)

    leaky_relu = layers.LeakyReLU()(batchnorm1)

    zero_pad2 = layers.ZeroPadding2D()(leaky_relu)  # (bs, 26, 16, 512)

    last = layers.Conv2D(1, 4, strides=1,
                         kernel_initializer=initializer)(zero_pad2)  # (bs, 23, 13, 1)

    return Model(inputs=[inp, tar], outputs=last)

discriminator = Discriminator(input_shape)



# compile and train
import tensorflow_addons as tfa

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + (100 * l1_loss)

    return total_gen_loss, gan_loss, l1_loss

def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

@tf.function
def train_step(input_image, target, epoch):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

    return {
        "gen_total_loss": gen_total_loss,
        "gen_gan_loss": gen_gan_loss,
        "gen_l1_loss": gen_l1_loss,
        "disc_loss": disc_loss,
    }

# To train the model, we would loop over the dataset, calling train_step() for each batch.





