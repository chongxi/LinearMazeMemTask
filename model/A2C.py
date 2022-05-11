from collections import deque 
import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class Replay_Buffer:
    def __init__(self, maxlen=100000, device='cpu'):
        self.buffer = deque(maxlen=maxlen)
        self.maxlen = maxlen
        self.device = device

    def store(self, s, a, r, s_, a_, d, episodic_mem):
        self.buffer.append([s, a, r, s_, a_, d, episodic_mem])

    def sample(self, batch_size):
        bat = random.sample(self.buffer, batch_size)
        batch = list(zip(*bat))
        data = []
        for i in range(len(batch)):
            data.append(torch.as_tensor(batch[i], dtype=torch.float32, device=self.device))
        return data

    def __len__(self):
        return len(self.buffer)


class A2C(nn.Module):
    """a MLP actor-critic network
    process: relu(Wx) -> pi, v
    Parameters
    ----------
    dim_input : int
        dim state space
    dim_hidden : int
        number of hidden units
    dim_output : int
        dim action space
    Attributes
    ----------
    ih : torch.nn.Linear
        input to hidden mapping
    actor : torch.nn.Linear
        the actor network
    critic : torch.nn.Linear
        the critic network
    _init_weights : helper func
        default weight init scheme
    """

    def __init__(self, dim_input, dim_hidden, dim_output):
        super(A2C, self).__init__()
        self.dim_input = dim_input
        self.dim_hidden = dim_hidden
        self.dim_output = dim_output
        self.ih = nn.Linear(dim_input, dim_hidden)
        self.actor = nn.Linear(dim_hidden, dim_output)
        self.critic = nn.Linear(dim_hidden, 1)
        # ortho_init(self)

    def forward(self, x, beta=1):
        """compute action distribution and value estimate, pi(a|s), v(s)
        Parameters
        ----------
        x : a vector
            a vector, state representation
        beta : float, >0
            softmax temp, big value -> more "randomness"
        Returns
        -------
        vector, scalar
            pi(a|s), v(s)
        """
        h = F.relu(self.ih(x))
        action_distribution = softmax(self.actor(h), beta)
        value_estimate = self.critic(h)
        return action_distribution, value_estimate

    def pick_action(self, action_distribution):
        """action selection by sampling from a multinomial.
        Parameters
        ----------
        action_distribution : 1d torch.tensor
            action distribution, pi(a|s)
        Returns
        -------
        torch.tensor(int), torch.tensor(float)
            sampled action, log_prob(sampled action)
        """
        m = torch.distributions.Categorical(action_distribution)
        a_t = m.sample()
        log_prob_a_t = m.log_prob(a_t)
        return a_t, log_prob_a_t


class A2C_linear(nn.Module):
    """a linear actor-critic network
    process: x -> pi, v
    Parameters
    ----------
    dim_input : int
        dim state space
    dim_output : int
        dim action space
    Attributes
    ----------
    actor : torch.nn.Linear
        the actor network
    critic : torch.nn.Linear
        the critic network
    """

    def __init__(self, dim_input, dim_output):
        super(A2C_linear, self).__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.actor = nn.Linear(dim_input, dim_output)
        self.critic = nn.Linear(dim_input, 1)

    def forward(self, x, beta=1):
        """compute action distribution and value estimate, pi(a|s), v(s)
        Parameters
        ----------
        x : a vector
            a vector, state representation
        beta : float, >0
            softmax temp, big value -> more "randomness"
        Returns
        -------
        vector, scalar
            pi(a|s), v(s)
        """
        action_distribution = softmax(self.actor(x), beta)
        value_estimate = self.critic(x)
        return action_distribution, value_estimate

    def pick_action(self, action_distribution):
        """action selection by sampling from a multinomial.
        Parameters
        ----------
        action_distribution : 1d torch.tensor
            action distribution, pi(a|s)
        Returns
        -------
        torch.tensor(int), torch.tensor(float)
            sampled action, log_prob(sampled action)
        """
        m = torch.distributions.Categorical(action_distribution)
        a_t = m.sample()
        log_prob_a_t = m.log_prob(a_t)
        return a_t, log_prob_a_t


class A2C_LSTM_EM(nn.Module):
    """a linear actor-critic network
    process: x -> pi, v
    Parameters
    ----------
    dim_input : int
        dim state space
    dim_output : int
        dim action space
    Attributes
    ----------
    actor : torch.nn.Linear
        the actor network
    critic : torch.nn.Linear
        the critic network
    """

    def __init__(self, dim_input, dim_hidden, dim_output, embedding_dim=512, episode_len=20):
        super(A2C_LSTM_EM, self).__init__()
        self.dim_input = dim_input
        self.dim_LSTM_out = dim_hidden
        self.dim_output = dim_output
        # self.actorInput = nn.Linear(dim_input, linear_input_dim)
        # self.actorInput  = nn.Sequential(   # input shape (1,1,45,45), output (1, 32, 4, 4), 32*4*4 = 512
        #     nn.Conv2d(1, 32, 5, stride=1, padding=2),
        #     nn.MaxPool2d(2, 2),
        #     nn.ReLU(),
        #     nn.Dropout2d(0.25),
        #     nn.Conv2d(32, 32, 5, stride=1, padding=1),
        #     nn.MaxPool2d(2, 2),
        #     nn.ReLU(),
        #     nn.Dropout2d(0.25),
        #     nn.Conv2d(32, 32, 5, stride=1, padding=1),
        #     nn.MaxPool2d(2, 2),
        #     nn.ReLU(),        
        # )
        self.actorLSTM = nn.LSTM(dim_input, self.dim_LSTM_out, 1) # (input_size, hidden_size, layers)
        self.actorOutput = nn.Sequential(nn.Linear(self.dim_LSTM_out, 128),
                                         nn.Linear(128, dim_output))
        self.critic = nn.Linear(self.dim_input, 1)
        self.layer_norm_1 = nn.LayerNorm((1,1,self.dim_LSTM_out))
        self.layer_norm_2 = nn.LayerNorm((1,1,self.dim_LSTM_out))
        self.replay_buffer = []
        self.episodic_length = episode_len
        self.episodic_Att = nn.Sequential(   #This neural network learns which episodic states to “attend” to and by how much.
            nn.Linear(dim_input, self.episodic_length),
            # nn.ReLU()
            # nn.Softmax(dim=0)
            # nn.Dropout(0.1)
        )
        self.position_LSTM = nn.LSTM(self.dim_LSTM_out, self.dim_LSTM_out)
        self.position_net = nn.Sequential(
            nn.Linear(60, 1),
            nn.ReLU()
        )
        self.reset_parameter()
        self.reset_episodic_mem()
        self.alpha = 0.99
        
    def reset_parameter(self):
        for name, wts in self.named_parameters():
            if 'weight' in name:
                torch.nn.init.orthogonal_(wts)
            elif 'bias' in name:
                torch.nn.init.constant_(wts, 0)
        
    def get_init_states(self, scale=.1):
        h_0 = torch.randn(1, 1, self.dim_LSTM_out) * scale # (num_layers, batch_size, hidden_size)
        c_0 = torch.randn(1, 1, self.dim_LSTM_out) * scale # (num_layers, batch_size, hidden_size)
        h_1 = torch.randn(1, 1, self.dim_LSTM_out) * scale # (num_layers, batch_size, hidden_size)
        c_1 = torch.randn(1, 1, self.dim_LSTM_out) * scale # (num_layers, batch_size, hidden_size)
        return h_0, c_0, h_1, c_1

    def init_episodic_mem(self, scale=.01):
        for i in range(self.episodic_length):
            _h = torch.randn(1, 1, self.dim_LSTM_out) * scale 
            # _c = torch.zeros(1, 1, self.dim_LSTM_out) 
            self.episodic_mem.append(_h.clone().detach())

    def forward(self, x, h1, c1, h2=None, c2=None, beta=1, allow_hidden_backprop=True):
        """compute action distribution and value estimate, pi(a|s), v(s)
        Parameters
        ----------
        x : a vector
            a vector, state representation
        beta : float, >0
            softmax temp, big value -> more "randomness"
        Returns
        -------
        vector, scalar
            pi(a|s), v(s)
        """
        # x = self.actorInput(x.view(1,1,45,45)).view(1, 1, -1) # CNN
        hidden, (new_h, new_c) = self.actorLSTM(x, (h1, c1))
        c = new_h + new_c
        c = c.view(1, 1, -1)
        if h2 is not None:
            # c = torch.cat((new_h, new_c), dim=-1)
            pos_hidden, (pos_h, pos_c) = self.position_LSTM(c, (h2, c2))
            self.pos_hidden = pos_hidden
            self.pos_h, self.pos_c = pos_h, pos_c
        # position_estimate = self.position_net(pos_hidden)
        em_out = self.read_episodic_mem(x).view(1,1,-1)
        out = self.alpha*em_out+(1-self.alpha)*(self.pos_hidden)
        # out = torch.cat((em_out, hidden), dim=-1)
        # out = self.layer_norm_1(out)
        output = self.actorOutput(out)
        # em_in = self.layer_norm_1(self.pos_hidden)  # self.pos_hidden+hidden
        em_in = hidden
        self.update_episodic_mem(em_in, allow_hidden_backprop)
        action_distribution = softmax(output, beta)
        value_estimate = self.critic(x)
        return action_distribution, value_estimate, new_h, new_c, self.pos_h, self.pos_c #, em_out

    def update_episodic_mem(self, h, allow_backprop=True):
        """
        Parameters
        ----------
        h : a vector
            a vector, current hidden state from LSTM
        """
        if allow_backprop:
            self.episodic_mem.append(h)  
            if len(self.episodic_mem) > self.episodic_length:
                self.episodic_mem.pop(0)
        else:
            self.episodic_mem.append(h.clone().detach())  
            if len(self.episodic_mem) > self.episodic_length:
                self.episodic_mem.pop(0)
        assert(len(self.episodic_mem) == self.episodic_length)

    def reset_episodic_mem(self, scale=0.01):
        self.episodic_mem = []
        self.init_episodic_mem(scale=scale)
    
    def read_episodic_mem(self, x):
        """
        Returns
        -------
        a vector
            a vector, a weighted hidden state from the episodic memory
        """
        w = self.episodic_Att(x).view(1, self.episodic_length)
        # w = F.softmax(w, 1)
        output = torch.mm(w, torch.stack(self.episodic_mem, dim=0).squeeze())
        # output = F.relu(output)
        return output

    def pick_action(self, action_distribution, noise_level=0):
        """action selection by sampling from a multinomial.
        Parameters
        ----------
        action_distribution : 1d torch.tensor
            action distribution, pi(a|s)
        Returns
        -------
        torch.tensor(int), torch.tensor(float)
            sampled action, log_prob(sampled action)
        """
        m = torch.distributions.Categorical(action_distribution)
        entropy = m.entropy().mean()
        if noise_level == 0:
            a_t = m.sample()
            log_prob_a_t = m.log_prob(a_t)
        elif noise_level > 0:
            noise = noise_level*torch.distributions.uniform.Uniform(low=0, high=1).sample((4,))
            # noisy_action_dist = F.softmax(action_distribution + noise, dim=0)
            noisy_action_dist = action_distribution + noise
            n = torch.distributions.Categorical(noisy_action_dist)
            a_t = n.sample()
            log_prob_a_t = m.log_prob(a_t)
        return a_t, log_prob_a_t, entropy

    def imprint(self, state, action, reward, next_state, done):
        """
        Parameters
        ----------
        state : a vector
            a vector, current state
        action : a vector
            a vector, action
        reward : scalar
            scalar, reward
        next_state : a vector
            a vector, next state
        done : bool
            bool, whether this is the last step
        """
        self.replay_buffer.append((state, action, reward, next_state, done, self.episodic_mem))


class A2C_ConvLSTM(nn.Module):
    def __init__(self, dim_obs, dim_hidden, dim_action, device='cpu'):
        super(A2C_ConvLSTM, self).__init__()
        self.dim_obs = dim_obs
        self.dim_action = dim_action
        self.device = device
        self.conv = nn.Sequential(
            nn.Conv2d(self.dim_obs, 32, 5, stride=1, padding=2),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 5, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(1024, 512, batch_first=True)
        self.act_net = nn.Sequential(
            nn.Linear(512, self.dim_action),
            # nn.Softmax(dim=0)
            # nn.ReLU(),
            # nn.Linear(128, self.dim_action),
        )
        self.cri_net = nn.Sequential(
            nn.Linear(512, 1),
            # nn.ReLU(),
            # nn.Linear(128, 1),
        )
        self.cx = torch.zeros(512).unsqueeze(0).unsqueeze(0).to(self.device)
        self.hx = torch.zeros(512).unsqueeze(0).unsqueeze(0).to(self.device)

    def forward(self, inputs):
        x = self.conv(inputs)
        x = x.view(-1, 1024).unsqueeze(0)

        x, (hx, cx) = self.lstm(x, (self.hx, self.cx))
        self.hx = hx
        self.cx = cx

        return self.act_net(x), self.cri_net(x)

    def reset_lstm(self):
        self.cx = torch.zeros(512).unsqueeze(0).unsqueeze(0).to(self.device)
        self.hx = torch.zeros(512).unsqueeze(0).unsqueeze(0).to(self.device)

    def pick_action(self, action_distribution):
        m = torch.distributions.Categorical(action_distribution)
        a_t = m.sample()
        log_prob_a_t = m.log_prob(a_t)
        return a_t, log_prob_a_t


def softmax(z, beta):
    """helper function, softmax with beta
    Parameters
    ----------
    z : torch tensor, has 1d underlying structure after torch.squeeze
        the raw logits
    beta : float, >0
        softmax temp, big value -> more "randomness"
    Returns
    -------
    1d torch tensor
        a probability distribution | beta
    """
    assert beta > 0
    return torch.nn.functional.softmax(torch.squeeze(z / beta), dim=0)
