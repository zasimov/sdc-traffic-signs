# German Signs Classifier

As a base architecture for my network I chosen LeNet-5 implementation from Udacity lessons. I decided to follow the next plan:

* try to learn LeNet-5 using original Udacity dataset. I got an accuracy 0.869 on validation set (epochs 10, batch size 128, learning rate 0.001)

* I added dropout (with keep probility 0.8) to prevent overfitting. I got accuracy 0.906 after 15 epochs.

* I added normalization (zero centration and std normalization). I got accuracy 0.93.

Next I decided to generate more data. I used standard rotation with random angle and color distortion (brightness and contrast). LeNet-5 with one dropout showed accuracy 0.94. I decided to add each one dropout (keep probability 0.8) and reach 0.95 accuracy. It's enough for this project =)

## Future work

* use GAN to generate more data
* "tune" LeNet-5 architecture (change number of layers and layers size)
* try to use grayscaled images
* try Stohastic Gradient Descend