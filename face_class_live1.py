# develop a classifier for the 5 Celebrity Faces Dataset
from random import choice
from numpy import load
from numpy import expand_dims
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from matplotlib import pyplot
import cv2
import main
import face
import numpy as np
from numpy import asarray
from mtcnn.mtcnn import MTCNN
from PIL import Image
from keras.models import load_model


def extract_face(image, required_size=(160, 160)):
	# load image from file
	# image = Image.open(filename)
	# convert to RGB, if needed
	# image = image.convert('RGB')
	# convert to array
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	pixels = asarray(image)
	# create the detector, using default weights
	detector = MTCNN()
	# detect faces in the image
	results = detector.detect_faces(pixels)
	print(results)
	# extract the bounding box from the first face
	x, m = [], []
	for j in results:
		x1, y1, width, height = j['box']
		# bug fix
		x1, y1 = abs(x1), abs(y1)
		x2, y2 = x1 + width, y1 + height
		# extract the face
		face = pixels[y1:y2, x1:x2]
		# resize pixels to the model size
		image = Image.fromarray(face)
		image = image.resize(required_size)
		face_array = asarray(image)
		x.append(face_array)
		m.append([x1, y1, x2, y2])
	# return face_array , [x1,y1,x2,y2]
	return x, m


def get_embedding(model, face_pixels):
	# scale pixel values
	# face_pixels = face_pixels.astype('float32')
	# standardize pixel values across channels (global)
	l = []
	for i in face_pixels:
		mean, std = i.mean(), i.std()
		i = (i - mean) / std
		# transform face into one sample
		samples = expand_dims(i, axis=0)
		# make prediction to get embedding
		yhat = model.predict(samples)
		l.append(yhat[0])
	# return yhat[0]
	return l


# v = cv2.VideoCapture(0)
# load faces
data = load('5-celebrity-faces-dataset.npz')
testX_faces = data['arr_2']

# load face embeddings
data = load('5-celebrity-faces-embeddings.npz')
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']

# normalize input vectors
in_encoder = Normalizer(norm='l2')
trainX = in_encoder.transform(trainX)
testX = in_encoder.transform(testX)

# label encode targets
out_encoder = LabelEncoder()
out_encoder.fit(trainy)
trainy = out_encoder.transform(trainy)
testy = out_encoder.transform(testy)

# fit model
model = SVC(kernel='linear', probability=True)
model.fit(trainX, trainy)
model1 = load_model('facenet_keras.h5')
frame = cv2.imread("group3.jpeg")
down_width = 1200
down_height = 800
down_points = (down_width, down_height)
frame = cv2.resize(frame, down_points, interpolation=cv2.INTER_LINEAR)

try:
	x, m = extract_face(frame)
	x_emb = get_embedding(model1, x)
# test model on a random example from the test dataset
	for i in range(len(x)):
		random_face_pixels = x[i]
		random_face_emb = x_emb[i]
		# random_face_class = testy[selection]
		# random_face_name = out_encoder.inverse_transform([random_face_class])

		# prediction for the face
		samples = expand_dims(random_face_emb, axis=0)
		yhat_class = model.predict(samples)
		yhat_prob = model.predict_proba(samples)

		# get name
		class_index = yhat_class[0]
		class_probability = yhat_prob[0, class_index] * 100
		predict_names = out_encoder.inverse_transform(yhat_class)
		print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
		cv2.rectangle(frame, (m[i][0], m[i][1]),
						(m[i][2], m[i][3]), (0, 255, 255), 2)
		font = cv2.FONT_HERSHEY_DUPLEX
		if class_probability > 99.99:
			cv2.putText(frame, predict_names[0], (m[i][0],m[i][1]), font, 1.0, (255, 255, 255), 1)
		else:
			cv2.putText(frame, 'Unknown', (m[i][0], m[i][1]), font, 1.0, (255, 255, 255), 1)
	cv2.imshow('frame', frame)
except:
	cv2.imshow('frame', frame)
# the 'q' button is set as the
# quitting button you may use any
# desired button of your choice

# v.release()
# cv2.destroyAllWindows()
while (cv2.waitKey(1) & 0xFF != ord('q')):
	pass
