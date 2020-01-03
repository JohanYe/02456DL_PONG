# Includes wrappers and VAE code
import gym
import torch.nn as nn
from collections import deque
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def _reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1) #pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env=None):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        super(EpisodicLifeEnv, self).__init__(env)
        self.lives = 0
        self.was_real_done = True
        self.was_real_reset = False

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert somtimes we stay in lives == 0 condtion for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def _reset(self):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset()
            self.was_real_reset = True
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
            self.was_real_reset = False
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class FireResetEnv(gym.Wrapper):
    def __init__(self, env=None):
        """For environments where the user need to press FIRE for the game to start."""
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def _reset(self):
        self.env.reset()
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset()
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset()
        return obs

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        """Return only every `skip`-th frame"""
        super(MaxAndSkipEnv, self).__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break

        max_frame = np.max(np.stack(self._obs_buffer), axis=0)

        return max_frame, total_reward, done, info

    def reset(self):
        """Clear past frame buffer and init. to first obs. from inner env."""
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs

def make_atari(env_id):
    env = gym.make(env_id)
    assert 'NoFrameskip' in env.spec.id
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = EpisodicLifeEnv(env)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    return env

# General stuff
def save_frames_as_gif(frames, filename=None):
    """
    Save a list of frames as a gif
    """
    patch = plt.imshow(frames[0])
    plt.axis('off')
    def animate(i):
        patch.set_data(frames[i])
    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    if filename:
        anim.save(filename, dpi=72, writer='imagemagick')

############## VAE STUFF ##############
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def dataloader_from_replay_mem(replay_memory,bs):
    X_train = np.array(replay_memory.memory)[:,0]
    X_train = np.stack(X_train)
    X_train = X_train[:,0,:,:].reshape(X_train.shape[0],1,X_train.shape[2],X_train.shape[3]) #Only using first frame in each framestack of 2
    X_train = torch.utils.data.DataLoader(torch.Tensor(X_train), batch_size=bs, shuffle=True)
    return X_train

class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()

    def forward(self, x):
        # Do your print / debug stuff here
        print(x.size())
        return x


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), size, 1, 1)


class VAE_preprocess(nn.Module):
    def __init__(self):
        super(VAE_preprocess, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=8, stride=3),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Conv2d(2, 3, 4, 2, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            # Flatten()
        )

        # self.fc_mu = nn.Linear(1604,196)
        # self.fc_logvar = nn.Linear(1604,196)
        # self.fc_back = nn.Linear(196,1604)

        self.fc_mu = nn.Conv2d(3, 3, 3, 1)
        self.fc_logvar = nn.Conv2d(3, 3, 3, 1)
        self.fc_back = nn.ConvTranspose2d(3, 3, 3, 1)

        self.decoder = nn.Sequential(
            # UnFlatten(),
            nn.ConvTranspose2d(3, 2, kernel_size=4, stride=2),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.ConvTranspose2d(2, 1, kernel_size=7, stride=3, padding=1),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        eps = torch.randn_like(mu)
        z = torch.exp(logvar * 0.5) * eps + mu
        return z

    def encode(self, x,stack = False):
        if stack: #if stacked frames, encode each frame separately and stack in end
            x1 = x[:,0].reshape(x.shape[0],1,80,80)
            x1,_,_ = self.encode(x1)
            x = x[:,1].reshape(x.shape[0],1,80,80)

        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)
        if stack:
            z = torch.cat((x1,z),1)
            #mu and logvar cannot be used if stacked frames.
            mu = None
            logvar = None
        return z, mu, logvar

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        # print(z.size())
        recon_x = self.fc_back(z)
        recon_x = self.decoder(recon_x)
        return recon_x, mu, logvar


def ELBO_loss(recon_x, x, mu, log_var, beta):
    # KL-divergence
    KLD = -1 / 2 * torch.sum(1 + log_var - mu.pow(2) - torch.exp(log_var), dim=1)

    # Reconstruction loss is a bit more difficult
    Reconstruction_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')

    return Reconstruction_loss / x.shape[0] + beta * torch.mean(KLD)

def train(VAE,data, optimizer,epochs = 10,beta=1):
    loss = []
    mean_loss = np.zeros(epochs)
    beta = 0
    for epoch in range(epochs):
        for idx, batch in enumerate(data):
            batch = batch.to(device)
            recon_x, mu, log_var = VAE(batch)
            batch_loss = ELBO_loss(recon_x,batch,mu,log_var,beta)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            loss.append(batch_loss.item())
        if beta < 1:
                beta += 0.005
        mean_loss[epoch] = np.mean(loss)
        loss = []
        if epoch % 5 == 0:
            print(epoch,mean_loss[epoch])
    return recon_x,batch,mean_loss,VAE

def show_recon_x(recon_x,batch):
    batch = batch.cpu().detach().numpy().reshape(batch.shape[0],80,80)
    recon_x = recon_x.cpu().detach().numpy().reshape(batch.shape[0],80,80)
    fig, axs = plt.subplots(4, 8,figsize=(15,15))
    for i in range(4):
        for j in range(4):
            axs[i,j].imshow(recon_x[4*i+j])
            axs[i,j+4].imshow(batch[4*i+j])
            axs[i,j].axis('off')
            axs[i,j+4].axis('off')
    plt.show()

def Wrap_modelstate(epoch,model,optimizer,epsilon,target_network,replay):
    Model_dict = {'epoch': epoch + 1,
                  'epsilon': epsilon,
                  'state_dict': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'target_network':target_network.state_dict(),
                  'replay_buffer':replay}
    return Model_dict