
import torch
import numpy as np
from torch.distributions import Categorical


class NCP_Sampler():
    """The Neural Clustering Process model for inference. 
    
    based on https://github.com/aripakman/neural_clustering_process
    """
    def __init__(self, model, data):
        """Load a trained NCP model and new spike waveform data
        Args:
            model: a trained model from trainer_model.NCP_trainer() 
            data: a torch array of shape (1, N, n_channels, n_timesteps) of spike waveforms
        """
        self.h_dim = model.params['h_dim']
        self.g_dim = model.params['g_dim']
        self.device = model.params['device']

        assert data.shape[0] == 1
        self.N = data.shape[1]

        data = data.to(self.device)
        data = data.view([self.N, data.shape[2], data.shape[3]])

        self.hs = model.h(data)
        self.qs = self.hs

        self.f = model.f
        self.g = model.g

    def sample(self, S, beam=False):
        """Posterior sampling of cluster assignments for each spike waveform

        Args:
            S: the number of parallel samples 
            beam: bool, whether to do beam search 
                if True: beam search with a beam size of S, keeping samples with the highest likelihood
                if False: random sample
        Returns:
            cs: a numpy array of shape (S, N) of S parallel samples of N cluster labels
            nll: a numpy array of shape (S,) of the negative log-likelihood of each sample
        """
        if beam:
            print("  Sampling with beam search.")

        assert type(S) == int
        if beam:
            beam_S = S
            S = 1

        cs = torch.zeros([S, self.N], dtype=torch.int64).to(self.device)
        previous_maxK = 1
        nll = torch.zeros(S).to(self.device)

        with torch.no_grad():
            for n in range(1, self.N):
                Ks, _ = cs.max(dim=1)
                Ks += 1
                maxK = Ks.max().item()
                minK = Ks.min().item()

                inds = {}
                for K in range(minK, maxK+1):
                    inds[K] = Ks == K

                if n == 1:
                    self.Q = self.qs[2:, :].sum(
                        dim=0).unsqueeze(0)  # [1, q_dim]
                    self.Hs = torch.zeros([S, 2, self.h_dim]).to(self.device)
                    self.Hs[:, 0, :] = self.hs[0, :]

                else:
                    if maxK > previous_maxK:
                        new_h = torch.zeros([S, 1, self.h_dim]).to(self.device)
                        self.Hs = torch.cat((self.Hs, new_h), dim=1)

                    self.Hs[np.arange(S), cs[:, n-1], :] += self.hs[n-1, :]

                    if n == self.N-1:
                        self.Q = torch.zeros([1, self.h_dim]).to(
                            self.device)  # [1, h_dim]
                    else:
                        self.Q[0, :] -= self.qs[n, :]

                previous_maxK = maxK

                if beam:
                    self.Hs = self.Hs[:, :maxK + 1]  # prune Hs in beam search 

                assert self.Hs.shape[1] == maxK + 1

                logprobs = torch.zeros([S, maxK+1]).to(self.device)
                rQ = self.Q.repeat(S, 1)
                rhn = self.hs[n, :].unsqueeze(0).repeat(S, 1)

                for k in range(maxK+1):
                    Hs2 = self.Hs.clone()
                    Hs2[:, k, :] += self.hs[n, :]

                    Hs2 = Hs2.view([S*(maxK+1), self.h_dim])
                    gs = self.g(Hs2).view([S, (maxK+1), self.g_dim])

                    for K in range(minK, maxK+1):
                        if k < K:
                            gs[inds[K], K:, :] = 0
                        elif k == K and K < maxK:
                            gs[inds[K], (K+1):, :] = 0

                    Gk = gs.sum(dim=1)

                    uu = torch.cat((Gk, rQ, rhn), dim=1)
                    logprobs[:, k] = torch.squeeze(self.f(uu))

                for K in range(minK, maxK):
                    logprobs[inds[K], K+1:] = float('-Inf')

                # Normalize log probability 
                m, _ = torch.max(logprobs, 1, keepdim=True)
                logprobs = logprobs - m - \
                    torch.log(torch.exp(logprobs-m).sum(dim=1, keepdim=True))

                if not beam:
                    probs = torch.exp(logprobs)
                    m = Categorical(probs)
                    ss = m.sample()
                    cs[:, n] = ss
                    nll -= logprobs[np.arange(S), ss]

                else:
                    new_nll = nll.unsqueeze(1) - logprobs  # [S, maxK+1]
                    n_row, n_col = new_nll.shape
                    beam_size = min(beam_S, n_row * n_col)
                    topk_nll, topk_idx = new_nll.reshape(
                        -1).topk(beam_size, largest=False)
                    col_idx = torch.arange(n_col).repeat(n_row, 1)
                    row_idx = torch.arange(n_row).repeat(n_col, 1).t()
                    col_topk = col_idx.reshape(-1)[topk_idx]
                    row_topk = row_idx.reshape(-1)[topk_idx]

                    cs = cs[row_topk, :]
                    cs[:, n] = col_topk
                    nll = topk_nll
                    self.Hs = self.Hs[row_topk]
                    S = beam_size

        cs, nll = cs.to('cpu').numpy(), nll.to('cpu').numpy()
        return cs, nll
