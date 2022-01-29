# A2C

A2C stands for Advantage Actor Critic (read the paper [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783v2)).

In policy gradient methods, we are trying to find the optimal policy function (a neural net outputs probability distribution of actions to take). In value based methods, we are trying to find the value network(s) (Q and V) and the optimal policy is just picking the action with maximum expected reward, i.e., max(Q(s, a)) in each state. In Actor-Critic methods, we train both policy network(s) and value network(s).

![](returns.png)

Using A2C, we see the agent gets max return by epoch 50. Time to reach epoch 50 (for Apple M1 8GB) is <5s. Make sure to install the required packages. Run `pip install -r requirements.txt` in your virtual environment. Train the agent by running `train.py`, and render gameplay by running `play.py`. 

`python3 play.py --render True --fps 45 --loc checkpoints/pi50.pt` renders the gameplay at 45 fps using the model @ checkpoints/pi50.pt...
