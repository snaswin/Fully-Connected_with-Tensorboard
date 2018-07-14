import numpy as np
import tensorflow as tf
import h5py
from tensorflow.examples.tutorials.mnist import input_data

#####Reading h5 file ###############################
hf = h5py.File('XY_50Nth_100N3th_split.h5','r')

#X_train = np.array(hf['X_train'])
#Y_train = np.array(hf['Y_train'])

#X_dev = np.array(hf['X_dev'])
#Y_dev = np.array(hf['Y_dev'])

#X_test = np.array(hf['X_test'])
#Y_test = np.array(hf['Y_test'])

##-rough
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
X_train = mnist.train.images
Y_train = mnist.train.labels

print(X_train.shape, Y_train.shape)



####################################################

def init_weights(shape, name):
	return tf.Variable(tf.random_normal(shape, stddev=0.01), name=name )
	
def model(X, w_h, w_h2, w_o, prob_keepinput, prob_keephidden):
	#Adding layer name scopes for better graph visualization
	with tf.name_scope("layer1"):
		X = tf.nn.dropout(X, prob_keepinput)
		h = tf.nn.relu(tf.matmul(X, w_h) )
	with tf.name_scope("layer2"):
		h = tf.nn.dropout(h, prob_keephidden)
		h2 = tf.nn.relu(tf.matmul(h, w_h2))
	with tf.name_scope("layer3"):
		h2 = tf.nn.dropout(h2, prob_keephidden)
		#return tf.nn.relu(tf.matmul(h2, w_o))
		return tf.matmul(h2, w_o)
		
		
#Step1- Gather input data and labels
# Done at the top

#Step2- Create Input and output placeholders for data
dim = X_train.shape[1]
outdim = Y_train.shape[1]
#dim = X_train[1] * X_train[2]
#outdim = Y_train[0]
X = tf.placeholder("float", [None, dim ], name="X")
Y = tf.placeholder("float", [None, outdim])

#Step3- Initialize weights
w_h = init_weights( (dim, int(dim/2)), name="w_h" )
w_h2 = init_weights( (int(dim/2), 625), name="w_h2")
w_o = init_weights( (625, outdim), name="w_o" )

#Step4- Add histogram summaries for weights
tf.summary.histogram("w_h_summary", w_h)
tf.summary.histogram("w_h2_summary", w_h2)
tf.summary.histogram("w_o_summary", w_o)

#Step5- Add Dropout to hidden & output layer
prob_keepinput = tf.placeholder("float", name="prob_keepinput")
prob_keephidden = tf.placeholder("float", name= "prob_keephidden")

#Step6- Create Model
mod = model(X, w_h, w_h2, w_o, prob_keepinput, prob_keephidden)

#Step7- Create cost func
with tf.name_scope("Cost"):
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=mod, labels=Y) )
	train_op = tf.train.RMSPropOptimizer(0.0001, 0.9).minimize(cost)
	#Add scalar cost summary for cost tensor
	tf.summary.scalar("cost", cost)
	
#Step8- Accuracy
with tf.name_scope("Accuracy"):
	correct_pred = tf.equal(tf.argmax(Y,1), tf.argmax(mod, 1) )
	acc = tf.reduce_mean(tf.cast(correct_pred, "float") ) #BOOL to Float
	#Add scalar summary for acc tensor
	tf.summary.scalar("Accu", acc)
	
#Step9- Create a session
with tf.Session().as_default() as sess:
	# Step10- Create a log writer. run 'tensorboard --logdir=./logs/nn_logs'
	writer = tf.summary.FileWriter("./logs/nn_logs", sess.graph)
	merged = tf.summary.merge_all()

#Step11- Initialize all variables
tf.global_variables_initializer().run(session=sess)

#Step12- Train model
epoch= 10000
for i in range(epoch):
	for start, end in zip(range(0, len(X_train), 128) , range(128, len(X_train)+1, 128)):
		sess.run(train_op, feed_dict={X: X_train[start:end], Y:Y_train[start:end], prob_keepinput: 0.8, prob_keephidden: 0.5})
	summary, accuracyout = sess.run([merged, acc], feed_dict={X: X_train, Y:Y_train, prob_keepinput:1.0, prob_keephidden:1.0})
	writer.add_summary(summary, i) #Write Summary
	print(i, accuracyout)



	
	
	
	
