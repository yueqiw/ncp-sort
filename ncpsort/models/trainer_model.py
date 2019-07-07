
'''
based on https://github.com/aripakman/neural_clustering_process
'''

import torch
from torch import nn
from ncpsort.models.spike_encoder import ResNet1DEncoder, BasicBlock
from ncpsort.utils.clustering import relabel


class NCP_Trainer(nn.Module):
    """The Neural CLustering Process model class for training.
    """
    def __init__(self, params):
        super(NCP_Trainer, self).__init__()

        self.params = params
        self.previous_n = 0
        self.previous_K = 1

        self.g_dim = params['g_dim']
        self.h_dim = params['h_dim']
        H = params['H_dim']

        self.device = params['device']

        self.h = ResNet1DEncoder(
            block=BasicBlock,
            num_blocks=params['resnet_blocks'],
            out_size=params['h_dim'],
            input_dim=params['n_channels'],
            planes=params['resnet_planes']
        )

        self.g = torch.nn.Sequential(
            torch.nn.Linear(self.h_dim, H),
            torch.nn.PReLU(),
            torch.nn.Linear(H, H),
            torch.nn.PReLU(),
            torch.nn.Linear(H, H),
            torch.nn.PReLU(),
            torch.nn.Linear(H, H),
            torch.nn.PReLU(),
            torch.nn.Linear(H, H),
            torch.nn.PReLU(),
            torch.nn.Linear(H, self.g_dim),
        )

        self.f = torch.nn.Sequential(
            torch.nn.Linear(self.g_dim + 2*self.h_dim, H),
            torch.nn.PReLU(),
            torch.nn.Linear(H, H),
            torch.nn.PReLU(),
            torch.nn.Linear(H, H),
            torch.nn.PReLU(),
            torch.nn.Linear(H, H),
            torch.nn.PReLU(),
            torch.nn.Linear(H, H),
            torch.nn.PReLU(),
            torch.nn.Linear(H, 1, bias=False),
        )

    def forward(self, data, cs, n):
        """Forward pass for the n-th element

        Elements 1,2,...,n-1 are already assigned 
        The n-th element is to be assigned. 
        Elements after the n-th are not assigned

        Args:
            data: A torch array of shape (batch_size, N_total, n_channels, n_timesteps)
            cs: A numpy array of assigned clusters 
            n: the index of the current element
        """
        assert(n == self.previous_n+1)
        self.previous_n = self.previous_n + 1

        K = len(set(cs[:n]))
        # the number of assigned clusters in [0:n]

        if n == 1:
            # first pass
            self.batch_size = data.shape[0]
            self.N = data.shape[1]
            assert (cs == relabel(cs)).all()

            data = data.to(self.device)
            data = data.view(
                [self.batch_size*self.N, data.shape[2], data.shape[3]])

            self.hs = self.h(data).view([self.batch_size, self.N, self.h_dim])
            self.Q = self.hs[:, 2:, ].sum(dim=1)     # [batch_size,h_dim]

            self.Hs = self.hs[:, 0, :].unsqueeze(1)
            gss = self.g(self.Hs.view([self.batch_size, self.h_dim]))
            self.gs = [[gss]]

            self.Hs = torch.cat((self.Hs, torch.zeros(
                [self.batch_size, 1, self.h_dim]).to(self.device)), dim=1)
            self.Hs = self.Hs + self.hs[:, 1, :].unsqueeze(1)
            self.new_gs = self.g(self.Hs.view(
                [self.batch_size*2, self.h_dim])).view([self.batch_size, 2, self.g_dim])
            self.Hs = self.Hs - self.hs[:, 1, :].unsqueeze(1)

        else:
            self.Hs[:, cs[n-1], :] = self.Hs[:, cs[n-1], :] + self.hs[:, n-1, :]

            if K == self.previous_K:
                self.gs[cs[n-1]].append(self.new_gs[:, cs[n-1], :])
            else:
                self.Hs = torch.cat((self.Hs, torch.zeros(
                    [self.batch_size, 1, self.h_dim]).to(self.device)), dim=1)
                self.gs.append([self.new_gs[:, cs[n-1], :]])

            self.Hs = self.Hs + self.hs[:, n, :].unsqueeze(1)
            self.new_gs = self.g(self.Hs.view(
                [self.batch_size*(K+1), self.h_dim])).view([self.batch_size, K+1, self.g_dim])
            self.Hs = self.Hs - self.hs[:, n, :].unsqueeze(1)

            if n == self.N-1:
                self.Q = torch.zeros([self.batch_size, self.h_dim]).to(self.device)  
                self.previous_n = 0

            else:
                self.Q = self.Q - self.hs[:, n, :]

        self.previous_K = K
        assert self.Hs.shape[1] == K+1
        # G = self.gs.sum(dim=1)   #[batch_size,g_dim]
        G = 0
        for k in range(K):
            G = G + self.gs[k][-1]

        uu = torch.zeros([self.batch_size, K+1, self.g_dim +
                          2*self.h_dim]).to(self.device)

        # loop over the K existing clusters for the n-th element to join
        for k in range(K):
            Gk = G - self.gs[k][-1] + self.new_gs[:, k, :]
            uu[:, k, :] = torch.cat((Gk, self.Q, self.hs[:, n, :]), dim=1)

        Gk = G + self.new_gs[:, K, :]
        uu[:, K, :] = torch.cat((Gk, self.Q, self.hs[:, n, :]), dim=1)

        logprobs = self.f(uu.view(
            [self.batch_size*(K+1), self.g_dim + 2*self.h_dim])).view([self.batch_size, K+1])

        # Normalize the log likelihood
        m, _ = torch.max(logprobs, 1, keepdim=True)        # [batch_size,1]
        logprobs = logprobs - m - \
            torch.log(torch.exp(logprobs-m).sum(dim=1, keepdim=True))

        return logprobs
