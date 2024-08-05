import torch
import numpy as np
from tqdm import tqdm

from oheflow.ohe_utils import ohe_to_categories

def pad_t_like_x(t, x):
    """Function to reshape the time vector t by the number of dimensions of x.

    Parameters
    ----------
    x : Tensor, shape (bs, *dim)
        represents the source minibatch
    t : FloatTensor, shape (bs)

    Returns
    -------
    t : Tensor, shape (bs, number of x dimensions)

    Example
    -------
    x: Tensor (bs, C, W, H)
    t: Vector (bs)
    pad_t_like_x(t, x): Tensor (bs, 1, 1, 1)
    """
    if isinstance(t, float):
        return t
    return t.reshape(-1, *([1] * (x.dim() - 1)))

@torch.no_grad()
def sample(model, num_samples, dim, N=None, solver='euler', device='cuda:0', use_tqdm=False):
    assert solver in ['euler', 'heun']
    tq = tqdm if use_tqdm else lambda x: x
    if N is None:
        N = model.cfm.N
    if solver == 'heun':
        N = (N + 1) // 2

    z0 = torch.randn([num_samples, dim], device=device)

    if model.cfm.pred_x1:
        pred_vt = lambda z_tmp, t_tmp: model.flow_net(z_tmp, t_tmp) - z0
    else:
        pred_vt = model.flow_net
    
    # traj = []
    dt = 1. / N
    z = z0.detach().clone()
    for i in tq(range(N)):
        t = torch.as_tensor(i / N).to(z.device)
        t_next = torch.as_tensor((i + 1) / N).to(z.device)
        vt = pred_vt(z, t)
        if solver == 'heun' and i < N-1:
            z_next = z.detach().clone() + vt * dt
            vt_next = pred_vt(z_next, t_next)
            vt = 0.5 * (vt + vt_next)
        
        z = z.detach().clone() + vt * dt
        # traj.append(z.detach().clone().unsqueeze(0))

    # post processing
    syn_num = z[:, :model.num_numerical_features]
    syn_cat = z[:, model.num_numerical_features:]
    
    z_ohe = torch.exp(syn_cat).round()
    syn_cat = ohe_to_categories(z_ohe, model.categories)

    syn_num = syn_num.cpu().numpy()
    syn_cat = syn_cat.cpu().numpy()

    # cast the output to the maximum if it is out of bound
    categories = np.array(model.categories).reshape(1, -1)
    categories = np.repeat(categories, syn_cat.shape[0], axis = 0) - 1

    print(f'casting rate: {(syn_cat > categories).sum()} / {syn_cat.shape[0] * syn_cat.shape[1]} = {(syn_cat > categories).mean()}')
    syn_cat[syn_cat > categories] = categories[syn_cat > categories]
    
    x_gen = np.concatenate([syn_num, syn_cat], axis=1)

    return x_gen

class BaseFlow():
    def __init__(self, pred_x1=False):
        self.N = 1000
        self.pred_x1 = pred_x1

    @torch.no_grad()
    def sample_ode_generative(self, z0, N, model, use_tqdm=False, solver='euler'):
        assert solver in ['euler', 'heun']
        tq = tqdm if use_tqdm else lambda x: x
        if N is None:
            N = self.N
        if solver == 'heun':
            N = (N + 1) // 2

        if self.pred_x1:
            pred_vt = lambda t_tmp, z_tmp: model(t_tmp, z_tmp) - z0
        else:
            pred_vt = model
        
        traj = []
        dt = 1. / N
        bs = z0.shape[0]
        z = z0.detach().clone()
        for i in tq(range(N)):
            t = torch.as_tensor(i / N).to(z.device)
            t_next = torch.as_tensor((i + 1) / N).to(z.device)
            vt = pred_vt(t, z)
            if solver == 'heun' and i < N-1:
                z_next = z.detach().clone() + vt * dt
                vt_next = pred_vt(t_next, z_next)
                vt = 0.5 * (vt + vt_next)
            
            z = z.detach().clone() + vt * dt
            traj.append(z.detach().clone().unsqueeze(0))

        return torch.cat(traj, 0)

class ConditionalFlowMatcher(BaseFlow):
    """Base class for conditional flow matching methods. This class implements the independent
    conditional flow matching methods from [1] and serves as a parent class for all other flow
    matching methods.

    It implements:
    - Drawing data from gaussian probability path N(t * x1 + (1 - t) * x0, sigma) function
    - conditional flow matching ut(x1|x0) = x1 - x0
    - score function $\nabla log p_t(x|x0, x1)$
    
    References
    ----------
    [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
    """

    def __init__(self, sigma: float=0.0, pred_x1: bool=False):
        r"""Initialize the ConditionalFlowMatcher class. It requires the hyper-parameter $\sigma$.

        Parameters
        ----------
        sigma : float
        """
        super().__init__(pred_x1)
        self.sigma = sigma

    def sample_noise_like(self, x):
        return torch.randn_like(x)

    def compute_mu_t(self, x0, x1, t):
        """
        Compute the mean of the probability path N(t * x1 + (1 - t) * x0, sigma), see (Eq.14) [1].

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the source minibatch
        t : FloatTensor, shape (bs)

        Returns
        -------
        mean mu_t: t * x1 + (1 - t) * x0

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        t = pad_t_like_x(t, x0)
        return t * x1 + (1 - t) * x0

    def compute_sigma_t(self, t):
        """
        Compute the standard deviation of the probability path N(t * x1 + (1 - t) * x0, sigma), see (Eq.14) [1].

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the source minibatch
        t : FloatTensor, shape (bs)

        Returns
        -------
        standard deviation sigma

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        del t
        return self.sigma

    def sample_xt(self, x0, x1, t, epsilon):
        """
        Draw a sample from the probability path N(t * x1 + (1 - t) * x0, sigma), see (Eq.14) [1].

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the source minibatch
        t : FloatTensor, shape (bs)
        epsilon : Tensor, shape (bs, *dim)
            noise sample from N(0, 1)

        Returns
        -------
        xt : Tensor, shape (bs, *dim)

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        mu_t = self.compute_mu_t(x0, x1, t)
        sigma_t = self.compute_sigma_t(t)
        sigma_t = pad_t_like_x(sigma_t, x0)
        return mu_t + sigma_t * epsilon

    def compute_conditional_flow(self, x0, x1, t, xt):
        """
        Compute the conditional vector field ut(x1|x0) = x1 - x0, see Eq.(15) [1].

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the source minibatch
        t : FloatTensor, shape (bs)
        xt : Tensor, shape (bs, *dim)
            represents the samples drawn from probability path pt

        Returns
        -------
        ut : conditional vector field ut(x1|x0) = x1 - x0

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        del t, xt
        return x1 - x0

    def sample_location_and_conditional_flow(self, x0, x1, return_noise=False):
        """
        Compute the sample xt (drawn from N(t * x1 + (1 - t) * x0, sigma))
        and the conditional vector field ut(x1|x0) = x1 - x0, see Eq.(15) [1].

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the source minibatch
        return_noise : bool
            return the noise sample epsilon

        Returns
        -------
        t : FloatTensor, shape (bs)
        xt : Tensor, shape (bs, *dim)
            represents the samples drawn from probability path pt
        ut : conditional vector field ut(x1|x0) = x1 - x0
        (optionally) eps: Tensor, shape (bs, *dim) such that xt = mu_t + sigma_t * epsilon

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        t = torch.rand(x0.shape[0]).type_as(x0)
        eps = self.sample_noise_like(x0)
        xt = self.sample_xt(x0, x1, t, eps)
        ut = self.compute_conditional_flow(x0, x1, t, xt)
        if self.pred_x1:
            ut = ut + x0    # ut = x1

        if return_noise:
            return t, xt, ut, eps
        else:
            return t, xt, ut
