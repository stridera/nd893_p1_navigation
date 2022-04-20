[//]: # "Image References"
[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Udacity Deep Reinforcement Learning Nanodegree

## Project 1: Navigation

This is the first project in the Udacity [Deep Reinforcement Learning Nanodegree.](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) The goal is to setup a deep RL model that will play the supplied Banana Environment.

### Getting Started

1. Checkout this repo.

2. Download the required environment.

   - [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
   - [Mac OSX](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
   - [Windows (32-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
   - [Windows (64-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

3. Unzip the file in a known location. The default location for the script is in `./env/` but you can provide your own path to the script.

4. Create and configure your virtual environment:

   1. Create and activate a new virtual environment.

      ```bash
      python3 -mvenv .venv
      source .venv/bin/activate
      ```

   2. Install dependencies:

      ```bash
      pip install python/
      ```

   3. (Optional) Train a new model. Warning: This will overwrite the supplied models. Note: Add `--help` to the command below to see the options and defaults.

      ```bash
      python train.py # Run with default options
      python train.py --env banana_env  --episodes 1000  --seed 0 # Run with given arguments.
      ```

   4. View existing models. Note: This command also supports the `--help` parameter to see options and defaults.

      ```bash
      python play.py # Runs the final.pth model with default params.
      python play.py --env banana_env --model_path models/solved.pth --fps 10 # Run the solved model at 10 frames per second.
      ```

      ​

### Environment Details

For this project, you will train an agent to navigate (and collect bananas!) in a large, square world.

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

### Trained Models

There are two models included. A final and solved model. The solved model was saved as soon as the model achieved the task of getting a score of +13 over 100 consecutive episodes. This happened after around 500 episodes. The final episode was trained until the curve flattened out, somewhere around 600 episodes with an average score around 16.

More information can be found in the [training report](Report.md).
