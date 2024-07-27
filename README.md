# Image-Recolouring-IITISoc
In the realm of digital imaging, the ability to breathe life into black and white photographs has long captivated researchers and artists alike. Our project delves into the fascinating world of image recoloring, exploring cutting-edge techniques to transform grayscale images into vibrant, full-color representations.
<br>
<br>
We experimented with models such as CycleGAN, which excels in unpaired image translation, and StyleGAN, known for its high-quality image generation. However, after rigorous testing and comparison, we ultimately settled on Pix2Pix as our GAN of choice for this project. Pix2Pix's ability to learn a mapping from input images to output images, coupled with its effectiveness in handling paired datasets, made it particularly well-suited for our grayscale to RGB colorization
<br>
<br>
The LAB color space played a crucial role in our recoloring process. Unlike RGB, LAB separates the luminance (lightness) channel 'L' from the color channels 'a' (green-red) and 'b' (blue-yellow). This separation allowed us to preserve the original image's luminance while focusing our colorization efforts on the 'a' and 'b' channels
<br>
We have found two ways of going through with LAB color space , have mentioned both of them codes in this repo too as LABmethod2.py and LAB Color Space.py
<br>
While we initially explored various Generative Adversarial Network (GAN) architectures, including CycleGAN and Pix2Pix, our research led us to DeOldify, a more advanced and specialized model for image colorization. DeOldify has proven to be superior to traditional GANs in several ways:
<br>
<br>
Stability: DeOldify employs NoGAN training, which combines the strengths of GANs with more stable training techniques, resulting in more consistent and higher-quality outputs.
<br>
Self-Attention: The incorporation of self-attention mechanisms allows DeOldify to better understand global context within an image, leading to more coherent and realistic colorization.
<br>
Perceptual Loss: DeOldify uses perceptual loss functions that align better with human visual perception, producing more natural-looking results.
<br>
Transfer Learning: By leveraging pre-trained networks, DeOldify can generalize well to a wide range of images, even with limited training data.
<br>
We have demonstrated how DeOldify's output surpasses that of traditional GANs in terms of color accuracy, consistency, and overall visual quality, marking a significant leap forward in the field of automatic image colorization.
<br>
<br>
CBIR---> Content Based Image Recolouring
<br>
This is realted to colouring the image with the context of the image using different types of Color Space 
<br>
Did this with the help of this Research Paper:- https://www.researchgate.net/publication/362688889_Comparative_Overview_of_Color_Models_for_Content-Based_Image_Retrieval

