import tensorflow as tf

class ResNet():
    def conv1d(self, x, filters, kernel_size, name, stride=1):
        return tf.layers.conv1d(x, filters, kernel_size, strides=stride, use_bias=True, padding='SAME',name=name)

    def resblock(self, x, filters, kernel_size, name, stride = 1, shortcut = False):
        resblock = self.conv1d(x, filters, kernel_size, stride=stride, name=name+'_conv1')
        shortcut_conv = tf.layers.batch_normalization(resblock, name=name+'_batch1')

        resblock = tf.nn.relu(shortcut_conv, name=name+'_relu')
        resblock = self.conv1d(resblock, filters, kernel_size, stride=stride, name=name+'_conv2')
        resblock = tf.layers.batch_normalization(resblock, name=name+'_batch2')

        if shortcut:
            return tf.nn.relu(resblock + shortcut)
        else:
            return tf.nn.relu(resblock + x)

    def count_model_params(self):
        total_parameters = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value

            #print(shape, variable_parameters)
            total_parameters += variable_parameters

        return total_parameters



class ResNetLogS(ResNet):
    def __init__(self, input_len):

        self.input_len = input_len

    def forward(self, X):
        X_reshaped=tf.reshape(X, shape=[-1, self.input_len, 1])

        conv1 = self.conv1d(X_reshaped, 9, 9, name='conv0', stride=1)

        # resblock1,2,3 (purple)
        resblock1 = self.resblock(conv1, 9, 9, name='resblock1')
        resblock2 = self.resblock(resblock1, 9, 9, name='resblock2')
        resblock3 = self.resblock(resblock2, 9, 9, name='resblock3')

        # resblock 4, 5, 6 (blue)
        resblock4 = self.resblock(resblock3, 18, 9, name='resblock4_s2', shortcut = True)
        resblock5 = self.resblock(resblock4, 18, 9, name='resblock5')
        resblock6 = self.resblock(resblock5, 18, 9, name='resblock6')

        # resblock 7, 8, 9 (yellow)
        resblock7 = self.resblock(resblock6, 36, 9, name='resblock7_s2', shortcut = True)
        resblock8 = self.resblock(resblock7, 36, 9, name='resblock8')
        resblock9 = self.resblock(resblock8, 36, 9, name='resblock9')

        # Flatten layer
        flatten = tf.layers.Flatten()(resblock9)

        # fully conected layer
        flcd1 = tf.layers.dense(flatten, 970, activation=tf.nn.relu)
        preds = tf.layers.dense(flcd1, 1, activation=None)

        return preds



class ResNetLogBB(ResNet):
    def __init__(self, input_len):
        
        self.input_len = input_len

    def forward(self, X):
        X_reshaped=tf.reshape(X, shape=[-1, self.input_len, 1])

        conv1 = self.conv1d(X_reshaped, 9, 9, name='conv1', stride=1)

        # resblock1,2 (purple)
        resblock1 = self.resblock(conv1, 9, 9, name='resblock1')
        resblock2 = self.resblock(resblock1, 9, 9, name='resblock2')

        # Flatten layer
        flatten = tf.layers.Flatten()(resblock2)

        # fully conected layer
        flcd1 = tf.layers.dense(flatten, 100, activation=tf.nn.relu)
        preds = tf.layers.dense(flcd1, 1, activation=None)

        return preds

