import numpy as np 

class MFM_generator():
    def __init__(self, Nmin, Nmax, maxK=None, poisson_lambda=2, dirichlet_alpha=1):
        assert Nmin > 0 and Nmax > Nmin 
        self.Nmin = Nmin 
        self.Nmax = Nmax 
        self.maxK = maxK 
        self.poisson_lambda = poisson_lambda
        self.dirichlet_alpha = dirichlet_alpha
    
    def generate(self, N=None):
        if N is None:
            N = np.random.randint(self.Nmin, self.Nmax)

        # K ~ Pk(k) = Poisson(k-1 | lambda), lambda > 0
        K_prior = np.random.poisson(self.poisson_lambda) + 1
        if self.maxK is not None:
            while K_prior > self.maxK:
                K_prior = np.random.poisson(self.poisson_lambda) + 1

        # (pi_1, ..., pi_k) ~ Dirichlet(alpha_1, ..., alpha_k)
        cluster_prob = np.random.dirichlet(np.ones(K_prior) * self.dirichlet_alpha)

        # Z_1, ..., Z_n ~ Cat(pi_1, ..., pi_k)
        cluster_assignments = np.random.choice(np.arange(1, K_prior+1), size=N, p=cluster_prob)
        # if some clusters have low prob, they may not show up in the samples. K_prior != K_sampled

        cluster_ids, cluster_count = np.unique(cluster_assignments, return_counts=True)
        K = len(cluster_count)
        clusters = np.zeros(N, dtype=int)
        for i in range(K):
            clusters[i+1] = cluster_count[i]
        return clusters, N, K
    
class CRP_generator():
    def __init__(self, Nmin, Nmax, maxK=None, alpha=0.7, no_ones=False):
        assert Nmin > 0 and Nmax > Nmin 
        self.Nmin = Nmin 
        self.Nmax = Nmax 
        self.maxK = maxK 
        self.alpha = alpha
        self.no_ones = no_ones
    
    def generate(self, N=None):
        keep = True
        while keep:
            if N is None:
                N = np.random.randint(self.Nmin, self.Nmax)
        
            clusters = np.zeros(N+2)  # 0...N,N+1  how many points in each cluster 
            clusters[0] = 0  # placeholder?
            clusters[1] = 1  # first cluster. we start filling the array here in order to use cumsum below 
            clusters[2] = self.alpha  # dispersion
            index_new = 2
            for n in range(N-1):     #we loop over N-1 particles because the first particle was assigned already to cluster[1]
                p = clusters/clusters.sum()
                z = np.argmax(np.random.multinomial(1,p))  # random draw from 0...n_clust+1
                if z < index_new:  # existing clusters 
                    clusters[z] +=1
                else:  # new cluster 
                    clusters[index_new] =1
                    index_new +=1
                    clusters[index_new] = self.alpha  # the next new cluster, alpha/(n_samples + alpha)
            
            clusters[index_new] = 0 
            clusters = clusters.astype(np.int32)  # list of int
            
            if self.no_ones:
                clusters = clusters[clusters!=1]
            N = int(np.sum(clusters))
            K = np.sum(clusters>0)
            keep = (self.maxK is not None and K > self.maxK)                       
            
        return clusters, N, K
