## Traffic experimentation
As a starting point I used a slightly modified network used for handwriting classification in the lecture:
- 1 2D convolutional layer of 16 3x3 filters
- 1 Max-Pooling layer with a pool size of 2x2
- 1 flattening layer
- 1 hidden dense layer of 128 neurons
- 1 dropout layer with a rate equal to 10%
- output layer

This configuration resulted in accuracy of 0.8626.

So I tried filter counts of 8 and 32. And surprisingly 8 filters actually resulted in better accuracy of 0.9312.

After that I tried different values of neurons in the hidden layer and while 64 resulted in terrible accuracy of 0.0545, increasing the count to 256 didn't positively affect the accuracy.

Changing the dropout rate in any direction resulted in worse results.

Then I experimented with adding 1 more convolutional layer and max-pooling layer and that resulted in more consistent training of good performing models.
Adding a second dense layer of 128 neurons increased that consistency even more.

The final configuration being:

- 2 2D convolutional layers of 16 3x3 filters
- 2 Max-Pooling layers with a pool size of 2x2
- 1 flattening layer
- 2 hidden dense layers of 128 neurons
- 1 dropout layer with a rate equal to 10%
- output layer

This configuration resulted in accuracy of:
* 0.9401 after 10 epochs
* 0.9487 after 20 epochs (best)
* 0.9190 after 50 epochs (overfitting)
