
Example use of MNIST data in Python:

First import the package:

>> import mnist_load_show as mnist

Then, load the data:

>> X , y = mnist.read_mnist_training_data();

There are 60,000 small (28-by-28) images of digits. Each such image is
scanned (pixel-by-pixel) into one row of X. Note: if you do not have
enough memory to read in all the digits, you can use

>> X, y = mnist.read_mnist_training_data( N );

where N is a smaller number of digits to read (for example 5000). To 
display an image, you can use:

>> mnist.visualize(X[5]);

To show the corresponding correct labels:

>> y[5]

In order to display a set of images (for example image 10 to 15):

>> mnist.visualize(X[10:16])

and the corresponding labels will be:

>> y[10:16]

