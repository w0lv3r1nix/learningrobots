from naoqi import *
import vision_definitions
import sys
from PIL import Image
from scipy.misc import imresize
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import model_from_json, Model
from keras.layers import *
from collections import deque
import imageio

def build_model():
	model_in = Input(shape=[60,80,1])
	mask_in_1 = Input(shape=[3])
	mask_in_2 = Input(shape=[3])
	# convolutional part
	conv1 = Conv2D(16,(3,3),activation='elu',padding='valid')(model_in)
	pool1 = MaxPooling2D((3,3),strides=(2,2))(conv1)
	conv2 = Conv2D(16,(3,3),activation='elu',padding='valid')(pool1)
	pool1 = MaxPooling2D((3,3),strides=(2,2))(conv2)
	conv3 = Conv2D(32,(3,3),activation='elu',padding='valid')(pool1)
	pool2 = MaxPooling2D((3,3),strides=(2,2))(conv3)
	conv4 = Conv2D(4,(1,1),activation='elu',padding='valid')(pool2)
	flat = Flatten()(conv4)
	cnn = Model(inputs=(model_in),outputs=(flat))
	# recurrent part
	tdis_in = Input(shape=(6,60,80,1))
	tdis = TimeDistributed(cnn)(tdis_in)
	fc1 = LSTM(8,activation='tanh')(tdis)
	# dueling part
	# value
	val_stream = Dense(32,activation='elu')(fc1)
	val_out = Dense(1,activation='linear')(val_stream)
	# advantage
	adv_stream = Dense(32,activation='elu')(fc1)
	adv_out1 = Dense(3,activation='linear',name='out1')(adv_stream)
	adv_out2 = Dense(3,activation='linear',name='out2')(adv_stream)
	adv_out_norm1 = Lambda(lambda a: a - K.mean(a),output_shape=[3])(adv_out1)
	adv_out_norm2 = Lambda(lambda a: a - K.mean(a),output_shape=[3])(adv_out2)
	# Q(s,a) = V(s) + A(s,a)
	out1 = Add()([adv_out_norm1,val_out])
	out2 = Add()([adv_out_norm2,val_out])
	# mask by actions
	mask_out_1 = Multiply()([out1,mask_in_1])
	mask_out_2 = Multiply()([out2,mask_in_2])
	return Model(inputs=(tdis_in,mask_in_1,mask_in_2),outputs=(mask_out_1,mask_out_2))

ROBOT_IP = "127.0.0.1"

# Loading the model does not work anymore, apparently because model was generated in python3
# and is loaded here with python2.7. Tried using same keras/tensorflow versions,
# but still doesn't work. No idea what changed between it working and not working.
# Workaround: build model from scratch.
# with open('model.json','r') as f:
# 	j = f.read()
# 	model = model_from_json(j)
model = build_model()
model.load_weights('model_weights.h5')
print('Loaded weights successfully.')

# buffer to store the last sample_rate (preprocessed) images
sample_rate = 6
state_buffer = deque()

headYawPos = .0
headPitchPos = .0

def render_target(radius=4):
    """
    generates and returns target marker
    values around target position are calculated as function of distance to center.
    center region is set to 0 to have more distinct features (edges) present
    """
    t = np.zeros([radius*2,radius*2])
    center = np.array([radius-.5,radius-.5])
    for i in range(radius*2):
        for j in range(radius*2):
            distance = np.abs(np.linalg.norm(center-np.array([i,j])))
            t[i,j] = np.clip((radius-distance)/radius,0,1)
    t[radius,radius] = 0
    t[radius-1,radius] = 0
    t[radius,radius-1] = 0
    t[radius-1,radius-1] = 0
    return t

def preprocess_image(image):
	""" Returns preprocessed image that resembles simulation observations. """
	image = np.array(image).astype(float)
	im = np.zeros(np.shape(image)[:2])
	# threshold each pixel for green channel
	for i in range(np.shape(im)[0]):
		for j in range(np.shape(im)[1]):
			# if the pixel is predominantly green, pixel = 1
			if image[i,j,1] - image[i,j,0] > 50 and image[i,j,1] - image[i,j,2] > 50:
				im[i,j] = 1
	# get median image position
	median = np.clip(np.round(np.median(np.where(im==1),1)).astype(int),0,1000)
	im *= 0
	im = im.astype(float)
	# preprocess image
	im[median[0]-4:median[0]+4,median[1]-4:median[1]+4] = render_target()
	return np.expand_dims(im,-1)

def main():
	"""
	Starts main control loop after connecting to NAO and setting up motion and camera proxys.
	Saves gif-'video' of loop.
	"""
	if len(sys.argv) < 2:
		nao_ip = ROBOT_IP
	else:
		nao_ip = sys.argv[1]

	headYawPos = .0
	headPitchPos = .0

	# motion proxy
	motion = ALProxy("ALMotion", nao_ip, 9559)

	# camera proxy, subscribe with respective parameters
	camProxy = ALProxy("ALVideoDevice", nao_ip, 9559)
	resolution = 7    # VGA
	colorSpace = 11   # RGB
	videoClient = camProxy.subscribeCamera("python_client", 0, resolution, colorSpace, 5)

	# for gif generation
	camImages, prepImages = [],[]

	# Set stiffness on for Head motors
	motion.setStiffnesses("Head", 1.0)

	print('Starting control loop.')
	for i in range(200):
		# get image from camera
		naoImage = camProxy.getImageRemote(videoClient)
		# sometimes image retrieval does not work
		if naoImage is not None:
			print('Image available.')
			# get height, width and actual image
			imageWidth = naoImage[0]
			imageHeight = naoImage[1]
			array = naoImage[6]

			# create a PIL image from pixel array and append to list for gif
			image = Image.frombytes("RGB", (imageWidth, imageHeight), array)
			camImages.append(np.array(image))
			# preprocess image and append to list for gif
			image = preprocess_image(image)
			prepImages.append(np.dstack([image*255]*3).astype(np.uint8))

			# fill buffer
			if len(state_buffer) >= sample_rate:
				state_buffer.popleft()
			state_buffer.append(np.expand_dims(image,0))

			# if enough images in buffer
			if len(state_buffer) == sample_rate:
				# set up input for network
				state = np.concatenate(state_buffer)
				# perform forward pass of network
				action_pitch,action_yaw = model.predict([np.expand_dims(state,0),np.ones([1,3]),np.ones([1,3])])
				# convert network output into action
				action_yaw = np.argmax(action_yaw)-1
				action_pitch = np.argmax(action_pitch)-1

				# perform movement in both directions by step_size in 100ms
				motion.post.angleInterpolation(["HeadYaw","HeadPitch"],[headYawPos-action_yaw*step_size,headPitchPos+action_pitch*step_size],[.1,.1],False)
				motion.waitUntilMoveIsFinished()
		else:
			# if image could not be fetched
			print('Image is None...')
	# save gif
	imageio.mimsave('cam.gif',camImages,duration=1/12)
	imageio.mimsave('prep.gif',prepImages,duration=1/12)
	# Gently set stiff off for Head motors
	motion.setStiffnesses("Head", 0.0)

if __name__ == "__main__":
	main()
