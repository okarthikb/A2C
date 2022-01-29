from train import *
from time import sleep
from torch import load, no_grad
from argparse import ArgumentParser


pi = Actor().to(device("cpu"))


def play(render, fps):
  d = False
  G = 0
  s = e.reset()
  seq = deque([zeros_like(s)] * (l - 1) + [s], maxlen=l)
  with no_grad():
    while not d:
      s = tensor(phi(seq), dtype=float32)
      i = pick(categorical(pi(s).squeeze()))
      s, r, d, _ = e.step(actions[i])
      seq.append(s)
      G += r
      if render:
        e.render()
        sleep(1 / fps)
  e.close()
  return G


if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("--render", type=bool, default=False, help="render the game")
  parser.add_argument("--fps", type=int, default=45, help="game fps")
  parser.add_argument("--loc", type=str, default=f"checkpoints/pi{epochs}.pt", help="location of model")
  args = parser.parse_args()
  
  pi.load_state_dict(load(args.loc))
  print(f"using model {args.loc}")
  print(f"return: {play(args.render, args.fps)}")
