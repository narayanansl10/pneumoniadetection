from keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image
from keras.preprocessing.image import load_img
import os


model=load_model("model_saved1.h5")

model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
images = []
pred = []
crct = 0	
folder = './chest_xray/val'
for filename in os.listdir(folder):
	try:
		imgx = mpimg.imread(os.path.join(folder, filename))
		img = Image.open(os.path.join(folder, filename)).convert('L')
		#img = load_img(os.path.join(folder, filename),grayscale="true")
		img = img.resize((128,128))
		img = np.reshape(img,[1,128,128,1])
	except:
		print('Cant import ' + filename)
		continue;
	classes = model.predict_classes(img)
	print(classes[0][0])
	print(filename)
	if(((filename.startswith('I') or filename.startswith('N')) and classes[0][0]==0) or (filename.startswith('p') and classes[0][0]==1)):
		crct+=1
	if img is not None:
		images.append(img)
#print(images)
#for image in os.listdir()
	#img = Image.open(image)
	#img = img.resize((224,224), Image.ANTIALIAS)
	#img = np.reshape(img,[1,224,224,3])
	#classes = model.predict_classes(img)
	#print(classes)
	#print(image)
print(str(crct*100/116) + "% Accuracy")