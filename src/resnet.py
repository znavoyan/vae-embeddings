import tensorflow as tf

class ResNet():
	def conv1d(self, x, filters, kernel_size, name, stride=1):
		return tf.layers.conv1d(x, filters, kernel_size, strides=stride, use_bias=True, padding='SAME',name=name)


	def resblock(self, x, filters, kernel_size, name):
		resblock=self.conv1d(x, filters, kernel_size, stride=1, name=name+'_conv1')
		resblock=tf.layers.batch_normalization(resblock, name=name+'_batch1')
		resblock=tf.nn.relu(resblock, name=name+'_relu')
		resblock=self.conv1d(resblock, filters, kernel_size, stride=1, name=name+'_conv2')
		resblock=tf.layers.batch_normalization(resblock, name=name+'_batch2')
		return tf.nn.relu(resblock+x)


	def resblock_shortcut(self, x, filters, kernel_size, name):
		resblock=self.conv1d(x, filters, kernel_size, stride=1, name=name+'_conv1')
		resblock=tf.layers.batch_normalization(resblock, name=name+'_batch1')
		resblock=tf.nn.relu(resblock, name=name+'_relu')
		resblock=self.conv1d(resblock, filters, kernel_size, stride=1, name=name+'_conv2')
		resblock=tf.layers.batch_normalization(resblock, name=name+'_batch2')

		shortcut_conv = self.conv1d(x, filters, 1, stride=1, name=name+'_sh_conv')
		return tf.nn.relu(resblock+shortcut_conv)


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



class ResNet20(ResNet):
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
		resblock4 = self.resblock_shortcut(resblock3, 18, 9, name='resblock4_s2')
		resblock5 = self.resblock(resblock4, 18, 9, name='resblock5')
		resblock6 = self.resblock(resblock5, 18, 9, name='resblock6')

		# resblock 7, 8, 9 (yellow)
		resblock7 = self.resblock_shortcut(resblock6, 36, 9, name='resblock7_s2')
		resblock8 = self.resblock(resblock7, 36, 9, name='resblock8')
		resblock9 = self.resblock(resblock8, 36, 9, name='resblock9')

		# Flatten layer
		flatten = tf.layers.Flatten()(resblock9)

		# fully conected layer
		flcd1 = tf.layers.dense(flatten, 970, activation=tf.nn.relu)
		preds = tf.layers.dense(flcd1, 1, activation=None)

		return preds



class ResNet6(ResNet):
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

