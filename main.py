import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # define tensor names
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    # restore model graph and weights
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    # load saved model
    default_graph = tf.get_default_graph()

    image_input = default_graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = default_graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = default_graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = default_graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = default_graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    
    return image_input, keep_prob, layer3_out, layer4_out, layer7_out
tests.test_load_vgg(load_vgg, tf)



def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes, dropout):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    # define initialisaton
    xavier_init = tf.contrib.layers.xavier_initializer()
    l2_regularizer = tf.contrib.layers.l2_regularizer(1e-4)

    # define convolutional layers with dropout and relu
    l7_conv_1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, padding='same',
                                kernel_initializer = xavier_init,
                                kernel_regularizer = l2_regularizer)
    l7_activation = tf.nn.relu(l7_conv_1x1)
    l7_dropout = tf.layers.dropout(l7_activation, rate = dropout)

    l4_conv_1x1 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, padding='same',
                                kernel_initializer = xavier_init,
                                kernel_regularizer = l2_regularizer)
    l4_activation = tf.nn.relu(l4_conv_1x1)
    l4_dropout = tf.layers.dropout(l4_activation, rate = dropout)

    l3_conv_1x1 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, padding='same',
                                kernel_initializer = xavier_init,
                                kernel_regularizer = l2_regularizer)
    l3_activation = tf.nn.relu(l3_conv_1x1)
    l3_dropout = tf.layers.dropout(l3_activation, rate = dropout)

    # Define fully convolutional layers
    # Upsample layer 7 and combine with layer 4
    fcn = tf.layers.conv2d_transpose(l7_dropout, num_classes, 4, 2, padding='same', kernel_initializer = xavier_init)
    fcn = tf.add(fcn, l4_dropout)

    # Upsample network and combine with layer 3
    fcn = tf.layers.conv2d_transpose(fcn, num_classes, 4, 2, padding='same', kernel_initializer = xavier_init)
    fcn = tf.add(fcn, l3_dropout)

    # Upsample network
    fcn = tf.layers.conv2d_transpose(fcn, num_classes, 16, 8, padding='same', kernel_initializer = xavier_init)

    return fcn
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # Define output and truth labels
    logits   = tf.reshape(nn_last_layer, (-1, num_classes), name='logits')
    ground_truth = tf.reshape(correct_label, (-1, num_classes))

    # Softmax with cross entropy loss
    softmax_logits = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels = ground_truth)
    cross_entropy_loss = tf.reduce_mean(softmax_logits)

    # Optimise using Adam
    optimiser = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss)

    return logits, optimiser, cross_entropy_loss
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, keep_prob_rate, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    print("Training...\n")

    for epoch in range(epochs):
        print("Epoch {} of {}...".format(epoch, epochs))

        for batch, (images, labels) in enumerate(get_batches_fn(batch_size)):
            feed_dict = { input_image: images, 
                          correct_label: labels, 
                          keep_prob: keep_prob_rate, 
                          learning_rate: 1e-4
                        }
            optim, loss = sess.run([train_op, cross_entropy_loss], feed_dict = feed_dict)

            print(" -> Batch: {}, Loss: {:.4f}".format(batch + 1, loss))

tests.test_train_nn(train_nn)


def run():
    
    # define vars
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    directory = "./model"

    # check model dir exists
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    model_path = os.path.join(directory, "SegNet")

    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Define vars
        epochs = 20
        batch_size = 2
        keep_prob_rate = 0.5

        # Define placeholders
        correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes], name='correct_label')
        learning_rate = tf.placeholder(tf.float32, [], name='learning_rate')

        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        print("Initializing network...")

        # Build FCN using load_vgg, layers, and optimize function
        (input, dropout, layer3, layer4, layer7) = load_vgg(sess, vgg_path)
        network = layers(layer3, layer4, layer7, num_classes, dropout)
        (logits, train_op, cross_entropy_loss) = optimize(network, correct_label, learning_rate, num_classes)
        
        print("done\n")

        # init op (after graph definition)
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        # define model saver
        saver = tf.train.Saver()

        # restore from checkpoint (if available)
        checkpoint = tf.train.latest_checkpoint(directory)
        if checkpoint:
            print("Restoring from checkpoint...", checkpoint)
            saver.restore(sess, checkpoint)
            print("Done.")
        else:
            print("Couldn't find checkpoint to restore from. Starting over.")

        train_params = helper.count_number_trainable_params()
        print("Total number of trainable parameters: {}M".format(round(train_params/1e+6, 2)))

        # Train FCN using the train_nn function
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, 
                 input, correct_label, dropout, keep_prob_rate, learning_rate)

        # OPTIONAL: Apply the trained model to a video

        # Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, dropout, input_image)


if __name__ == '__main__':
    run()
