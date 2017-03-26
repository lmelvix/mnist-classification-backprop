import numpy
import activation_function as af
import read_mnist
import logging

class NeuralNetwork(object):

	def __init__(self):
		self.train_image_filename = 'train_image.npy'
		self.train_label_filename = 'train_label.npy'
		self.valid_image_filename = 'valid_image.npy'
		self.valid_label_filename = 'valid_label.npy'
		self.test_image_filename = 'test_image.npy'
		self.test_label_filename = 'test_label.npy'

	def load_train_data(self):
		'''
		Load Training and Validation set for 1-10 digit MNIST sample processed and stored in numpy file.
		:return: Training and Validation images and labels
		'''
		train_images = numpy.load(self.train_image_filename)
		train_label = numpy.load(self.train_label_filename)
		valid_images = numpy.load(self.valid_image_filename)
		valid_label = numpy.load(self.valid_label_filename)
		test_images = numpy.load(self.test_image_filename)
		test_label = numpy.load(self.test_label_filename)
		return train_images, train_label, valid_images, valid_label, test_images, test_label

	def get_train_data(self, input_digit):

		# Extract Train data
		input_image, input_label = read_mnist.load_mnist('training', digits=input_digit)
		input_image = self.preprocess_image(input_image)
		input_image = numpy.insert(input_image, 0, 1, axis=1)

		validation_image = input_image[-int(len(input_image)*0.20):]
		validation_label = input_label[-int(len(input_label)*0.20):]
		train_image = input_image[:int(len(input_image)*0.80)]
		train_label = input_label[:int(len(input_label)*0.80)]

		numpy.save(self.train_image_filename, train_image)
		numpy.save(self.train_label_filename, train_label)
		numpy.save(self.valid_image_filename, validation_image)
		numpy.save(self.valid_label_filename, validation_label)

		# Extract Test data
		input_image, input_label = read_mnist.load_mnist('testing', digits=input_digit)
		input_image = self.preprocess_image(input_image)
		input_image = numpy.insert(input_image, 0, 1, axis=1)

		test_image = input_image
		test_label = input_label

		numpy.save(self.test_image_filename, test_image)
		numpy.save(self.test_label_filename, test_label)
		return train_image, train_label, validation_image, validation_label, test_image, test_label

	@staticmethod
	def preprocess_image(train_image):
		preprocess_image_set = []
		for image in train_image:
			image = [float(x/255.) for x in image]
			image_mean = numpy.mean(image)
			image = [x - image_mean for x in image]
			preprocess_image_set.append(image)
		return preprocess_image_set

	def one_hot_encoder(self, train_label, sample_size):
		one_hot_matrix = numpy.zeros((sample_size, 10), dtype=numpy.int)
		for idx in xrange(len(train_label)):
			one_hot_matrix[idx][train_label[idx]-1] = 1
		return one_hot_matrix


def training():

	input_set = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
	load_data = True
	layer1_size = 100
	layer2_size = 10
	regularize = 0.001
	layer1_learningrate = 0.00001
	layer2_learningrate = 0.00001
	max_epochs = 1000
	batch_size = 128
	prev_err = -1
	prev_err2 = -1
	momentum = 0.9

	# Get training images as a matrix
	bck_propagate = NeuralNetwork()

	if load_data:
		all_train_images, all_train_label, all_validation_images, \
		all_validation_label, all_test_images, all_test_label = bck_propagate.load_train_data()
	else:
		all_train_images, all_train_label, all_validation_images, \
		all_validation_label, all_test_images, all_test_label = bck_propagate.get_train_data(input_set)

	logging.info("Extracting images and labels")

	# Uncomment to work with single sample
	train_images = all_train_images[1:2]
	train_label = all_train_label[1:2]
	validation_images = all_validation_images[1:1]
	validation_label = all_validation_label[1:1]


	# Split into Training and Validation Data
	sample_size, feature_size = all_train_images.shape
	test_sample_size, test_feature_size = all_test_images.shape
	validation_sample_size, validation_feature_size = all_validation_images.shape
	num_batch = int(sample_size/batch_size)

	# Convert training label to one-hot-encoding format
	all_one_hot_train_label = bck_propagate.one_hot_encoder(all_train_label, sample_size)
	all_one_hot_validation_label = bck_propagate.one_hot_encoder(all_validation_label, validation_sample_size)
	all_one_hot_test_label = bck_propagate.one_hot_encoder(all_test_label, test_sample_size)
	logging.info("Converted labels to one-hot encoding format")

	# Send training images to layer-1 for Logistic
	activ_fn = af.ActivationFunction(feature_size, sample_size, layer1_size, layer2_size)
	w1, w2 = activ_fn.initialize_weights()
	old_weight_1, old_weight_2 = w1, w2
	best_weight_1, best_weight_2 = w1, w2
	logging.info("Initialized weights")

	m_itr = 1
	for _ in xrange(max_epochs):

		# Zip train images and labels together for shuffling
		zipped_train = zip(all_train_images, all_one_hot_train_label)
		numpy.random.shuffle(zipped_train)
		mini_batches = numpy.vsplit(numpy.squeeze(zipped_train), num_batch)

		# iterate mini-batches for training
		for batch in mini_batches:

			# Extract feature set and labels of the batch for training
			train_images, one_hot_train_label = zip(*batch)
			train_images = numpy.asmatrix(train_images)
			one_hot_train_label = numpy.asmatrix(one_hot_train_label)

			# Forward Propagation
			# layer1_output = activ_fn.logistic_regression(train_images, w1)

			layer1_output = activ_fn.tanh(w1, train_images)
			logging.info("Computed Layer-1")
			layer2_output = activ_fn.softmax_regression(layer1_output, w2)

			# print "Weight1 : ", w1.shape
			# print "Input : ", train_images.shape
			# print "Weight2 : ", w2.shape
			# print "TanH : ", layer1_output.shape
			# print "Output : ", layer2_output.shape
			# print "----------------------------------"
			logging.info("Computed Layer-2")

			# Compute error at the layer-2
			output_error = one_hot_train_label - layer2_output

			# Update Layer-2 weight backpropagating error and perform gradient descent
			updated_w2 = activ_fn.gradient_update(output_error, w2, layer2_learningrate,
			                                      layer1_output, regularize, momentum)
			logging.info("Backpropagated Layer-2 weight")

			# Compute gradient for Layer-1
			hidden_gradient = w2 * output_error.T
			# propagating_gradient = activ_fn.logistic_gradient(w1, train_images)
			propagating_gradient = activ_fn.tanh_gradient(w1, train_images)
			hidden_gradient = numpy.multiply(hidden_gradient.T, propagating_gradient)

			# Update Layer-1 weight backpropagating error and perform gradient descent
			updated_w1 = activ_fn.gradient_update(hidden_gradient, w1, layer1_learningrate,
			                                      train_images, regularize, momentum)
			logging.info("Backpropagated Layer-1 weight")

			# Update weights for next iteration
			w1 = updated_w1
			w2 = updated_w2

		# Validate updated network
		v_error = testing(all_validation_images, all_one_hot_validation_label, w1, w2, layer1_size, layer2_size)
		error = v_error
		print str(m_itr) + " : " + str(error*100) + "%"
		m_itr += 1

		# Keep track of error from last three epochs and stop when error starts increasing
		if (prev_err2 < prev_err) and (prev_err < error):
			print "\n\nBEST VALIDATION ACCURACY : " + str((1 - error)*100) + "%"
			break
		prev_err2 = prev_err
		prev_err = error
		best_weight_1 = old_weight_1
		best_weight_2 = old_weight_2
		old_weight_1 = best_weight_1
		old_weight_2 = best_weight_2

	# Compute Accuracy on Test set
	test_error = testing(all_test_images, all_one_hot_test_label, best_weight_1, best_weight_2, layer1_size, layer2_size)
	error = test_error
	print "TEST ACCURACY : " + str((1 - error)*100) + "%"



def testing(test_image, test_label, weight_1, weight_2, layer1_size, layer2_size):
	sample_size, feature_size = test_image.shape
	activ_fn = af.ActivationFunction(feature_size, sample_size, layer1_size, layer2_size)

	# Forward Propagation
	# layer1_output = activ_fn.logistic_regression(test_image, weight_1)
	layer1_output = activ_fn.tanh(weight_1, test_image)
	layer2_output = activ_fn.softmax_regression(layer1_output, weight_2)

	# Threshold output to 1 for max element in each row
	thresholded_output = activ_fn.threshold_output(layer2_output)

	# Compute error in prediction
	error = numpy.sum(numpy.absolute(thresholded_output - test_label), axis=1) / 2
	error_percentage = float(numpy.sum(error, axis=0)) / float(error.shape[0])
	# print str(float(numpy.sum(error, axis=0))) + " / " + str(float(error.shape[0]))
	return error_percentage


if __name__ == "__main__":
	training()