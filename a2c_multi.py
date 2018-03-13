from multiprocessing import Pipe, Process
from skimage.color import rgb2gray
from skimage.transform import resize
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
import gym
import os


parser = argparse.ArgumentParser()
parser.add_argument('--env_name', default='PongDeterministic-v4', type=str,
                    help='gym environment')
parser.add_argument('--lr', default=3e-4, type=float,
                    help='learning rate')
parser.add_argument('--gamma', default=0.99, type=float,
                    help='discount factor')
parser.add_argument('--n_workers', default=24, type=int,
                    help='number of process')
parser.add_argument('--n_step', default=20, type=int,
                    help='number of max step for update')
parser.add_argument('--render', default=False, type=bool,
                    help='render the environment')
parser.add_argument('--cuda', default=True, type=bool,
                    help='use GPU or not')
parser.add_argument('--save_dir', default='./models', type=str,
                    help='directory of model save')
parser.add_argument('--save_interval', default=5000, type=int,
                    help='interval of saving model')
args = parser.parse_args()


def pre_processing(obs):
    processed_obs = np.uint8(
        resize(rgb2gray(obs), (84, 84), mode='constant')*255)
    return processed_obs


def worker(remote, parent_remote, env):
    parent_remote.close()

    score = 0
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            ob, reward, done, info = env.step(data)
            if args.render:
                env.render()
            score += reward

            if done:
                print("Score: {}".format(score))
                score = 0
                env.reset()
                ob, *_ = env.step(1)

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
        self.avg_max_prob = []
        if torch.cuda.is_available() and args.cuda:
            self.net.cuda()
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        else:
            try:
                self.net = torch.load(os.path.join(args.save_dir, args.env_name))
            except:
                print("No saved files")

    def get_actions(self, obses):
        obses = np.float32(obses / 255.0)
        obses = Variable(torch.from_numpy(obses))
        if torch.cuda.is_available() and args.cuda:
            obses = obses.cuda()
        probs, value = self.net(obses)
        probs = probs.data.cpu().numpy()

        avg_max_probs = np.mean(np.max(probs, axis=-1))
        self.avg_max_prob.append(avg_max_probs)

        acts = [np.nonzero(np.random.multinomial(1, prob))[0] for prob in probs]
        return np.array(acts)

    def train(self, obs, rews, dones, acts):
        obs = np.float32(obs / 255.0).reshape(-1, 4, 84, 84)
        obs = Variable(torch.from_numpy(obs))

        if torch.cuda.is_available() and args.cuda:
            obs = obs.cuda()
        a, v = self.net(obs)

        v = v.view(args.n_step, args.n_workers)
        a = a.view(args.n_step, args.n_workers, -1)

        discounted_rews = self.discounted_rewards(rews, dones, v[-1].data.cpu().numpy())
        discounted_rews -= np.mean(discounted_rews)
        discounted_rews /= np.std(discounted_rews) + 1e-10

        discounted_rews = Variable(torch.from_numpy(discounted_rews))\
                                .type(torch.FloatTensor)
        acts = torch.from_numpy(acts)\
                                .type(torch.LongTensor)

        acts_onehot = torch.zeros(a.size()).type(torch.FloatTensor)
        acts_onehot = Variable(acts_onehot.scatter_(-1, acts, 1))

        if torch.cuda.is_available() and args.cuda:
            discounted_rews = discounted_rews.cuda()
            acts_onehot = acts_onehot.cuda()
        
        ## critic loss
        v_loss = (v - discounted_rews).pow(2).mean()

        ## actor loss
        good_prob = ((a * acts_onehot).sum(-1) + 1e-10).log()
        entropy = -(a * (a + 1e-10).log()).mean()
        a_loss = -(good_prob * (discounted_rews - v.detach())).mean() + 0.01*entropy

        loss = 0.5*v_loss + a_loss

        self.optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm(self.net.parameters(), 30)
        self.optimizer.step()

    def discounted_rewards(self, rewses, doneses, v):
        discounted_rewses = np.zeros_like(rewses)
        running_add = v

        for t in reversed(range(0, len(rewses))):
            running_add[doneses[t]] = 0
            running_add = running_add * args.gamma + rewses[t]
            discounted_rewses[t] = running_add
        return discounted_rewses

    def run(self):
        t = 0
        obs = self.envs.reset()
        obses = obs
        state = np.stack([obses, obses, obses, obses], axis=1)
        global_step = 0
        avg_max_prob_plt = []
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
            global_step += 1
            if t % args.n_step == 0:
                self.train(states, rewses,
                           doneses, actses)
                t = 0

            if global_step % 100 == 0:
                avg_max_prob_plt.append(np.mean(self.avg_max_prob))

            if global_step % args.save_interval == 0:
                torch.save(self.net, os.path.join(args.save_dir, args.env_name))
                plt.plot(avg_max_prob_plt)
                plt.savefig("./avg_max_prob_plt.jpg")


if __name__ == "__main__":
    agent = Agent(env_name=args.env_name,
                  n_workers=args.n_workers)
    agent.run()
