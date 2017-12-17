import os
import matplotlib.pyplot as plt
import scipy.io as spio
import numpy as np 
import tensorflow as tf

DATA_FILENAME = 'c1p8.mat'

# omit tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

sess = tf.InteractiveSession()

data = spio.loadmat(DATA_FILENAME, squeeze_me=True)

# stim is the stimulus sequence
stim = tf.constant(data['stim'], tf.float64, name="stim")

# rho is the spike counts
rho = tf.constant(data['rho'], tf.float64, name="rho")

# temporal kernelSize of 150 corresponds to 300 milliseconds
# each time step is 2 ms
kernelSize = tf.constant(150, name="kernelSize")

totalCount = tf.reduce_sum(rho, name="totalCount")

# sta = Spike-Triggered Average
sta = tf.Variable(tf.zeros([kernelSize], tf.float64), 
	dtype=tf.float64, name="sta")

# stc = Spike-Triggered Covariance
# stc = tf.Variable(tf.zeros([kernelSize, kernelSize], tf.float64), 
# 	dtype=tf.float64, name="sta")

# init i to kernelSize + 1
i = tf.add(kernelSize, 1, name="i")

# loop till i < len(rho)
sta_loop_cond = lambda i, sta: tf.less(i, tf.size(rho))

# loop body
# rho[i] * stim[i - kernelSize + 1 : i]
def sta_loop_body(index, sta):
	rho_i = tf.gather(rho, index, name="rho_i")
	window_indices = tf.range(index - kernelSize + 1, index + 1)
	stim_window = tf.gather(stim, window_indices, name="stim_window")
	tmp = tf.scalar_mul(rho_i, stim_window)

	return (index + 1, tf.add(sta, tmp))

sta = tf.while_loop(sta_loop_cond, sta_loop_body, (i, sta))[1] # extract sta from tuple
sta = tf.divide(sta, totalCount)

# STC is not quite working yet

# -------------------------------------------------------

# Attempt 1 for STC
# spike_indices = tf.where(tf.greater(rho, 0.5))
# spike_stim = tf.gather(stim, spike_indices)
# stc_constant = tf.divide(1, tf.subtract(totalCount, 1))
# stc = tf.scalar_mul(stc_constant, tf.matmul(spike_stim, tf.transpose(spike_stim)))

# -------------------------------------------------------

# Attempt 2 for STC
# stc_loop_cond = lambda i, stc: tf.less(i, tf.size(rho))
# def stc_loop_body(i, stc):
# 	rho_i = tf.gather(rho, i)
# 	stim_normalized = tf.subtract(tf.gather(stim, i), sta)
# 	tmp = tf.scalar_mul(rho_i, stim_normalized)
# 	tmp = tf.multiply(tmp, tf.transpose(stim_normalized))
# 	return (i + 1, tf.add(stc, tmp))

# i = tf.constant(1)
# stc = tf.while_loop(stc_loop_cond, stc_loop_body, (i, sta))[1] # extract sta from tuple
# stc = tf.divide(stc, stc_constant)

writer = tf.summary.FileWriter('./graphs', sess.graph)

sess.run(tf.global_variables_initializer())

plt.plot(sta.eval())
plt.xlabel('Time (ms)')
plt.ylabel('Spike-Triggered Average')
plt.title('Spike-Triggered Average of the H1 Neuron')

plt.savefig("./graphs/sta.png")

writer.close()
