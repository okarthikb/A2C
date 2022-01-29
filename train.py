import gym
import matplotlib.pyplot as plt
from time import time
from collections import deque
from argparse import ArgumentParser
from numpy import array, zeros_like, expand_dims
from torch import tensor, float32, save, device, dot, log, stack
from torch.optim import Adam
from torch.nn import Module, Conv2d, Flatten, Linear
from torch.nn.functional import relu, softmax
from torch.distributions import Categorical


name = "CartPole-v0"  # "Pong-v0"
e = gym.make(name)
actions = [0, 1]  # [2, 3]


class Actor(Module):
  def __init__(self):
    super(Actor, self).__init__()
    """
    self.conv1 = Conv2d(1, 16, 4, 4) 
    self.conv2 = Conv2d(16, 32, 4, 4)
    self.flatten = Flatten()
    self.out = Linear(800, len(actions))
    """
    self.fc = Linear(4, 512)
    self.out = Linear(512, len(actions))

  def forward(self, x):
    """
    x = relu(self.conv1(x))
    x = relu(self.conv2(x))
    x = relu(self.flatten(x))
    """
    x = relu(self.fc(x))
    x = softmax(self.out(x), -1)
    return x


class Critic(Module):
  def __init__(self):
    super(Critic, self).__init__()
    """
    self.conv1 = Conv2d(1, 16, 4, 4) 
    self.conv2 = Conv2d(16, 32, 4, 4)
    self.flatten = Flatten()
    self.out = Linear(800, 1)
    """
    self.fc = Linear(4, 512)
    self.out = Linear(512, 1)

  def forward(self, x):
    """
    x = relu(self.conv1(x))
    x = relu(self.conv2(x))
    x = relu(self.flatten(x))
    """
    x = relu(self.fc(x))
    x = self.out(x)
    return x


def phi(seq):
  seq = array(seq)
  """
  seq = seq[::2, 34:194:2, ::2, 1]
  seq[seq == 72] = 0
  seq[seq != 0] = 1
  seq = seq[1] - seq[0]
  seq = expand_dims([seq], 0)
  """
  return seq


pi = Actor().to(device("cpu"))
V = Critic().to(device("cpu"))


epochs = 200
N = 5
l = 1  # 3
gamma = 0.9999
mod = 10
lr = 5e-3

opt = Adam(list(pi.parameters()) + list(V.parameters()), lr)


def categorical(p):
  return Categorical(p)


def pick(logits):
  return logits.sample().item()


def accumulate(rs):
  Gs = [r for r in rs]
  for t in range(len(rs) - 2, -1, -1):
    Gs[t] += gamma * Gs[t + 1]
  return array(Gs)


def computeLoss(Vs, logps, rs):
  Vs, logps = stack(Vs), stack(logps)
  Gs = tensor(accumulate(rs), dtype=float32)
  As = Gs - Vs
  return dot(As, As) - dot(As.detach(), logps)


def tau():
  d = False
  Vs, logps, rs = [], [], []
  s = e.reset()
  seq = deque([zeros_like(s)] * (l - 1) + [s], maxlen=l)
  while not d:
    s = tensor(phi(seq), dtype=float32)
    p, v = pi(s).squeeze(), V(s).squeeze()
    i = pick(categorical(p))
    s, r, d, _ = e.step(actions[i])
    seq.append(s)
    Vs.append(v)
    logps.append(log(p[i]))
    rs.append(r)
  return Vs, logps, rs


def train():
  returns, losses = [], []
  for epoch in range(1, epochs + 1):
    T = [tau() for _ in range(N)]
    loss = sum(computeLoss(Vs, logps, rs) for Vs, logps, rs in T) / N
    opt.zero_grad()
    loss.backward()
    opt.step()
    returns.append(sum(sum(rs) for _, _, rs in T) / N)
    if epoch % mod == 0:
      print("epoch: {}\treturn: {:.2f}".format(epoch, returns[-1]))
      save(pi.state_dict(), f"checkpoints/pi{epoch}.pt")
  return returns


if __name__ == "__main__":
  returns = train()
  fig = plt.figure(figsize=(8, 8))
  plt.plot(list(range(1, epochs + 1)), returns)
  plt.savefig("checkpoints/returns.png")
