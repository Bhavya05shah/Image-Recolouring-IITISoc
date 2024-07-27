import cv2 as cv
import numpy as np

#copied the models from the github repo of richzhang
protoxt_path = 'Models/colorization_deploy_v2.prototxt'#specifies the neural network
model_path = 'Models/colorization_release_v2.caffemodel'#contains the trained weights and parameters
kernel_path = 'models/pts_in_hull.npy'#helps in storing predefined colors and also mapping RGB to LAB
img_path = 'Photos/bw.jpg'

# Loaded the pre-trained network 
network = cv.dnn.readNetFromCaffe(protoxt_path, model_path)
points = np.load(kernel_path)

# Process the points(?)
points = points.transpose().reshape(2, 313, 1, 1)
network.getLayer(network.getLayerId("class8_ab")).blobs = [points.astype(np.float32)]
network.getLayer(network.getLayerId('conv8_313_rh')).blobs = [np.full([1, 313], 2.606, np.float32)]

# Load and process the black and white image
bw_img = cv.imread(img_path)
normalized = bw_img.astype("float32") / 255.0 #converting the values from 0 to 255 --> 0 to 1
lab = cv.cvtColor(normalized, cv.COLOR_BGR2LAB)#converting the image to LAB color space

# Resize the L channel and subtract 50
resized = cv.resize(lab, (224, 224))
L = cv.split(resized)[0]#splitting the resized image into l,a,b and storing the l image 
L -= 50# for reducing the brightness

# Set the input for the network and forward pass(?)
network.setInput(cv.dnn.blobFromImage(L))
ab = network.forward()[0, :, :, :].transpose((1, 2, 0))

# Resizing the ab channels and mix with the L channel to create the LAB image
ab = cv.resize(ab, (bw_img.shape[1], bw_img.shape[0]))
L = cv.split(lab)[0]

colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
colorized = cv.cvtColor(colorized, cv.COLOR_LAB2BGR)
colorized = (255 * colorized).astype("uint8")

# Display the original black and white and the colorized images
cv.imshow("BW Image", bw_img)
cv.imshow("Colorized", colorized)
cv.waitKey(0)