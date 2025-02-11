import cv2 as cv
import numpy as np

#Models from the github repo of richzhang
protoxt_path = 'Models/colorization_deploy_v2.prototxt'#specifies the neural network
model_path = 'Models/colorization_release_v2.caffemodel'#contains the trained weights and parameters
kernel_path = 'models/pts_in_hull.npy'#helps in storing predefined colors and also mapping RGB to LAB , contains color clusters
img_path = 'Photos/bw.jpg'

#Loaded the pre-trained network 
network = cv.dnn.readNetFromCaffe(protoxt_path, model_path)
points = np.load(kernel_path)

#Process the points(?)
#this 313 means that there are 313 color clusters and reshaped from(2,313) to this where (1,1) is to make it an easier fit for the network
points = points.transpose().reshape(2, 313, 1, 1)#transposes (313,2) to (2,313)
#class8_ab is the layer from the network getting extracted and points/clusters are being assigned to the blobs(which stores the weights)
network.getLayer(network.getLayerId("class8_ab")).blobs = [points.astype(np.float32)]#the float32 is the required format for this network

#here another network layer operating at the end which is responsible for refining the predictions of the ab channels
#a numpy array where there is a single output for each of the 313 color clusters and a bias of 2.606 (which is an experimentally determined value)
#gets added to the each of 313 raw scores from this layer and then gets converted to probabilities
network.getLayer(network.getLayerId('conv8_313_rh')).blobs = [np.full([1, 313], 2.606, np.float32)]

# Load and process the black and white image
bw_img = cv.imread(img_path)
normalized = bw_img.astype("float32") / 255.0 #converting the values from 0 to 255 --> 0 to 1
lab = cv.cvtColor(normalized, cv.COLOR_BGR2LAB)#converting the image to LAB color space

# Resize the L channel and subtract 50
resized = cv.resize(lab, (224, 224))#here the model is designed so to have an input of 224x224 pixel image
L = cv.split(resized)[0]#splitting the resized image into l,a,b and storing the l image 
L -= 50# for reducing the brightness and centered around 0

# Set the input for the network and forward pass(?)
network.setInput(cv.dnn.blobFromImage(L))#It reshapes the image to a 4D array of shape (N->no. of images, C->no. of channels, H, W) 
#to make this a suitable input to the deep learning model and blob just stores the data
ab = network.forward()[0, :, :, :].transpose((1, 2, 0))#the output is a 3D array of shape(2,224,224) and gets coverted to (224,224,2)

#Resizing the ab channels and mix with the L channel to create the LAB image
ab = cv.resize(ab, (bw_img.shape[1], bw_img.shape[0]))
L = cv.split(lab)[0]

#adding new axis here for the ab channel as the L is a 2D array and the LAB is a 3D(H,W,3) 
colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
colorized = cv.cvtColor(colorized, cv.COLOR_LAB2BGR)
colorized = (255 * colorized).astype("uint8")#due to normalization the values were between 0 to 1 

cv.imshow("BW Image", bw_img)
cv.imshow("Colorized", colorized)
cv.waitKey(0)
