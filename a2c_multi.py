from multiprocessing import Pipe, Process
from skimage.color import rgb2gray
from skimage.transform import resize
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np
import argparse
import ppaquette_gym_super_mario
import gym


parser = argparse.ArgumentParser()
parser.add_argument('--env_name', default='BreakoutDeterministic-v4', type=str,
                    help='gym environment')
parser.add_argument('--lr', default=3e-4, type=float,
                    help='learning rate')
parser.add_argument('--gamma', default=0.99, type=float,
                    help='discount factor')
parser.add_argument('--n_workers', default=4, type=int,
                    help='number of process')
parser.add_argument('--n_step', default=20, type=int,
                    help='number of max step for update')
parser.add_argument('--cuda', default=True, type=bool,
                    help='use GPU or not')
args = parser.parse_args()


def pre_processing(obs):
    processed_obs = np.uint8(
        resize(rgb2gray(obs), (84, 84), mode='constant')*255)
    return processed_obs


def worker(remote, parent_remote, env):
    parent_remote.close()

    start_life = 5
    score = 0
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            ob, reward, done, info = env.step(data)
            score += reward
            if start_life > info['ale.lives']:
                reward = -1
                start_life = info['ale.lives']

            if done:
                print("Score: {}".format(score))
                score = 0
                ob = env.reset()

            ob = pre_processing(ob)
            remote.send((ob, reward, done, info))

        if cmd == 'reset':
            ob = env.reset()
            ob = pre_processing(ob)
            remote.send(ob)

        if cmd == 'close':
            remote.close()
            break

        if cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))

        else:
            NotImplementedError


class EnvWrapper:
    def __init__(self, envs):
        nenvs = len(envs)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, env))
                for (work_remote, remote, env) in zip(self.work_remotes, self.remotes, envs)]

        for p in self.ps:
            p.daemon = True
            p.start()

        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        self.observation_space, self.action_space = self.remotes[0].recv()

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
          remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
          remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()


class ATARInet(torch.nn.Module):
    def __init__(self, output_size):
        super(ATARInet, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=4,
                           out_channels=16, 
                           kernel_size=8, 
                           stride=4)
        self.conv2 = torch.nn.Conv2d(in_channels=16,
                           out_channels=32,
                           kernel_size=4,
                           stride=2)
        self.conv3 = torch.nn.Conv2d(in_channels=32,
                           out_channels=32,
                           kernel_size=3,
                           stride=1)
        self.a_fc = torch.nn.Linear(1568, 256)
        self.v_fc = torch.nn.Linear(1568, 256)

        self.a_out = torch.nn.Linear(256, output_size)
        self.v_out = torch.nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 224*7)
        a = F.relu(self.a_fc(x))
        a = F.softmax(self.a_out(a), dim=-1)
        v = F.relu(self.v_fc(x))
        v = self.v_out(v)

        return a, v


class Agent:
    def __init__(self, env_name, n_workers):
        self.envs = [gym.make(env_name) for _ in range(n_workers)]
        self.envs = EnvWrapper(self.envs)
        self.action_size = self.envs.action_space.n
        self.net = ATARInet(self.action_size)
        self.optimizer = optim.Adam(self.net.parameters(), lr=args.lr)
        if torch.cuda.is_available() and args.cuda:
            self.net.cuda()

    def get_actions(self, obses):
        obses = np.float32(obses / 255.0)
        obses = Variable(torch.from_numpy(obses))
        if torch.cuda.is_available() and args.cuda:
            obses = obses.cuda()
        probs, value = self.net(obses)
        probs = probs.data.cpu().numpy()

        acts = []
        for prob in probs:
            act = np.random.choice(self.action_size, 1, p=prob)
            acts.append(act)
        return np.array(acts)

    def train(self, obs, rews, dones, acts):
        discounted_rews = self.discounted_rewards(rews, dones)
        discounted_rews -= np.mean(discounted_rews)
        discounted_rews /= np.std(discounted_rews) + 1e-8

        discounted_rews = Variable(torch.from_numpy(discounted_rews))\
                                .type(torch.FloatTensor)
        acts = torch.from_numpy(acts)\
                                .type(torch.LongTensor)

        obs = np.float32(obs / 255.0).reshape(-1, 4, 84, 84)
        obs = Variable(torch.from_numpy(obs))
        a, v = self.net(obs)

        v = v.view(20, 4)
        a = a.view(20, 4, -1)

        acts_onehot = torch.zeros(a.size()).type(torch.FloatTensor)
        acts_onehot = Variable(acts_onehot.scatter_(-1, acts, 1))

        ## critic loss
        v_loss = (v - discounted_rews).pow(2).mean()

        ## actor loss
        good_prob = ((a * acts_onehot).sum(-1) + 1e-10)
        a_loss = -(good_prob.log() * (v.detach() - discounted_rews)).sum()

        loss = 0.5*v_loss + a_loss

        self.optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm(self.net.parameters(), 40)
        self.optimizer.step()


    def discounted_rewards(self, rewses, doneses):
        discounted_rewses = np.zeros_like(rewses)
        running_add = np.zeros_like(rewses[0])

        for t in reversed(range(0, len(rewses))):
            ##TODO: fix running add if game over
            for i in range(args.n_workers):
                if doneses[t][i]:
                    running_add[i] = 0
            running_add = running_add * args.gamma + rewses[t]
            discounted_rewses[t] = running_add
        return discounted_rewses

    def run(self):
        t = 0
        obs = self.envs.reset()
        obses = obs
        state = np.stack([obses, obses, obses, obses], axis=1)
        while True:
            acts = self.get_actions(state)
            self.envs.step_async(acts)
            obs, rews, dones, _ = self.envs.step_wait()
            rews = np.clip(rews, -1., 1.)
            next_state = np.append(state[:, 1:, :, :], obs[:, None], axis=1)

            if t == 0:
                states = state[None]
                rewses = rews[None]
                doneses = dones[None]
                actses = acts[None]
            else:
                states = np.concatenate((states, state[None]))
                rewses = np.concatenate((rewses, rews[None]))
                doneses = np.concatenate((doneses, dones[None]))
                actses = np.concatenate((actses, acts[None]))

            state = next_state

            t += 1
            if t % args.n_step == 0:
                self.train(states, rewses,
                           doneses, actses)
                t = 0


if __name__ == "__main__":
    agent = Agent(env_name=args.env_name,
                  n_workers=args.n_workers)
    agent.run()
