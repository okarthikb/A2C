# A2C

A2C stands for Advantage Actor Critic. In policy gradient methods, we are trying to find the optimal policy function (a neural net outputs probability distribution of actions to take). In value based methods, we are trying to find the value network(s) (Q and V) and the optimal policy is just picking the action with maximum expected reward, i.e., max(Q(s, a)) in each state. In Actor-Critic methods, we train both policy network(s) and value network(s).

![returns.png]

Using A2C, we see the agent gets max return by epoch 50. Time to reach epoch 50 (for Apple M1 8GB) is ~5s.
