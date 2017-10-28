# German Signs Classifier

I chose LeNet-5 implementation from Udacity lessons as a base architecture for my network. I decided to follow this plan:

* try to train LeNet-5 using original Udacity dataset: As a result, I got the accuracy 0.869 on validation set (epochs 10, batch size 128, learning rate 0.001)

* add dropouts to prevent overfitting: I added one dropout with keep probability 0.8. As a result, I got accuracy 0.906 after 15 epochs.

* add normalization (center around mean and normalize using std). As a result, I got accuracy 0.93.

Then I decided to generate more data. I used rotation with a random angle and color distortion (brightness and contrast). LeNet-5 with one dropout showed the result of 0.94 accuracy. I decided to add one more dropout (with keep probability 0.8), and the model reached 0.95 accuracy. It's enough for this project =)

## Future work

* use GAN to generate more data

* "tune" LeNet-5 architecture (change number of layers and layers size)

* try to use grayscaled images

* try Stohastic Gradient Descend instead of Adam optimizer