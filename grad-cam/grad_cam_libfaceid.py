import numpy as np
import caffe
import cv2

# Load the deploy prototxt and caffemodel
deploy_file = 'libfaceid/models/estimation/gender_deploy.prototxt'
model_file = 'libfaceid/models/estimation/gender_net.caffemodel'
net = caffe.Net(deploy_file, model_file, caffe.TEST)

def preprocess_image(image_path):
    # Load the image
    image = caffe.io.load_image(image_path)
    # Resize the image to match the input size of the network
    image = caffe.io.resize_image(image, net.blobs['data'].data.shape[-2:])
    # Subtract the mean image (if available)
    if 'mean' in net.transformer.__dict__.keys():
        mean = net.transformer.mean['data']
        image = image - mean
    # Transpose the image to match the Caffe's input blob shape
    image = image.transpose((2, 0, 1))
    # Add an extra dimension to represent batch size (as we have only one image)
    image = np.expand_dims(image, axis=0)
    return image

def forward_pass(image):
    # Set the image as the input blob of the network
    net.blobs['data'].data[...] = image
    # Forward pass
    net.forward()

def compute_gradient(output_layer):
    # Get the output blob of the specified layer
    output_blob = net.blobs[output_layer]
    # Clear any previous gradients
    net.zero_grad()
    # Set the gradient of the output blob to be 1 for the predicted class
    output_blob.diff[...] = 0
    predicted_class = np.argmax(output_blob.data)
    output_blob.diff[0, predicted_class] = 1
    # Backward pass to compute gradients
    net.backward(start=output_layer)

def compute_grad_cam(feature_layer):
    # Get the feature blob and gradient blob of the specified layer
    feature_blob = net.blobs[feature_layer]
    gradient_blob = net.blobs[feature_layer + '_grad']
    # Compute the global average pooling of the gradients
    weights = np.mean(gradient_blob.data, axis=(2, 3))
    # Compute the weighted combination of the feature maps
    grad_cam = np.sum(weights * feature_blob.data, axis=1)
    grad_cam = np.maximum(grad_cam, 0)  # ReLU activation
    return grad_cam

def visualize_grad_cam(image, grad_cam):
    # Resize the Grad-CAM heatmap to match the original image size
    grad_cam = cv2.resize(grad_cam[0], image.shape[1::-1])
    # Normalize the heatmap values between 0 and 1
    grad_cam = (grad_cam - np.min(grad_cam)) / (np.max(grad_cam) - np.min(grad_cam))
    # Convert the heatmap to a color map using a jet colormap
    heatmap = cv2.applyColorMap(np.uint8(255 * grad_cam), cv2.COLORMAP_JET)
    # Overlay

if __name__ == '__main__':
    # Load and preprocess the input image
    image_path = "A Chaturvedi.jpg"
    image = preprocess_image(image_path)

    # Forward pass through the network
    forward_pass(image)

    # Specify the output layer and feature layer for Grad-CAM
    output_layer = 'fc8'
    feature_layer = 'conv3'

    # Compute the gradient and Grad-CAM
    compute_gradient(output_layer)
    grad_cam = compute_grad_cam(feature_layer)

    # Visualize Grad-CAM
    original_image = cv2.imread(image_path)
    visualize_grad_cam(original_image, grad_cam)
