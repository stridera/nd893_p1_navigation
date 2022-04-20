[//]: # "Image References"
[scores]: images/score.png "Training Score Graph"

# Training Report

## Double Deep Q Network Model

The model selected stayed true to the Double Deep Q Network model shown in the class. The approach used two QNetworks and an experience replay buffer to select actions from.

### Model Architecture

The QNetworks followed the paper with 3 linear layers linked with ReLU (Rectified Linear Unit) activation functions.

The model input and output were obtained directly from the environment. In this case, we have 37 inputs, and 4 outputs.

So the final trained model network looks like this:

```python
Linear(37, 64)
ReLU()
Linear(64, 64)
ReLU()
Linear(64, 4)
```

We use an Adam optimizer to train the models and a mean square function for determining loss.

### Hyper-parameters

The model was trained using the following hyper-parameters:

- Buffer Size: `10,000` We keep up to 10,000 samples around to train the model with.
- Batch Size: `64` Choosing a batch size is difficult. 64 seems like a good place for memory efficiency and training speed.
- Gamma: `0.99` The gamma factor is used to determine the reward discount. This will slowly discount the reward to make it so the chain of actions leading toward a positive reward are recognized.
- Epsilon:
  - Start: `1.0` We start using completely random movements. Since our model has not seen anything yet, it makes since that we just skip it for now.
  - End: `0.01` At the end, we still want to do some random movements. An epsilon of `0.01` means we do a random movement 1% of the time.
  - Decay: `0.995` We lower the epsilon from start to end by increments of `0.995`. This is slow enough to allow enough random movements at the start but starts to use more of the model as it's trained.
- Tau: `0.001` The tau value is used to soft update the target network. This means we only slowly update the target network using the following update schedule: *θ*−=*θ*×*τ*+*θ*−×(1−*τ*)
- Learning Rate: `0.0005` By keeping the learning rate really small, we can have our model slowly adjust the weights as it gets a good reward. While this environment may allow us to learn faster, it is probably not worth it since it might overshoot an optimal weight.

## Results

The model trained quickly, reaching a solved status (+13/100eps) around 400-500 episodes in. When we allowed the model to continue, it would generally improve slightly more and peak at around +16/100 eps.

![Scores Graph][scores]

## Future Improvements

For a simple environment, I can't imagine that we would get much improvement even with the edition of some more advanced model features. While these added features would let us train faster, the model already reaches a solved state in only 10-20 minutes.

If we want to look at improving our score, we would probably want to start looking at a better reward strategy. Perhaps penalize negatives more to make it really want to avoid the blue bananas and focus more on the yellow ones.
