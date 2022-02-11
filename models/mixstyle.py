import random
from contextlib import contextmanager
import torch
import torch.nn as nn


def deactivate_mixstyle(m):
    print('Dectivating mixstyle')
    if type(m) == MixStyle:
        m.set_activation_status(False)


def activate_mixstyle(m):
    print('Activating mixstyle')
    if type(m) == MixStyle:
        m.set_activation_status(True)

def update_mixing_coeff(m, lamda):
    if type(m) == MixStyle:
        m.lamda=lamda
    print(f'Updating mixing coeff to {m.lamda}')

def random_mixstyle(m):
    if type(m) == MixStyle:
        m.update_mix_method('random')

def sourcedomain_mixstyle(m):
    print('Applying sourcedomain mixstyle')
    if type(m) == MixStyle:
        m.update_mix_method('sourcedomain')

def crossdomain_mixstyle(m):
    print('Applying crossdomain mixstyle')
    if type(m) == MixStyle:
        m.update_mix_method('crossdomain')


@contextmanager
def run_without_mixstyle(model):
    # Assume MixStyle was initially activated
    try:
        model.apply(deactivate_mixstyle)
        yield
    finally:
        model.apply(activate_mixstyle)


@contextmanager
def run_with_mixstyle(model, mix=None):
    # Assume MixStyle was initially deactivated
    if mix == 'random':
        model.apply(random_mixstyle)

    elif mix == 'crossdomain':
        model.apply(crossdomain_mixstyle)

    try:
        model.apply(activate_mixstyle)
        yield
    finally:
        model.apply(deactivate_mixstyle)


class MixStyle(nn.Module):
    """MixStyle.
    Reference:
      Zhou et al. Domain Generalization with MixStyle. ICLR 2021.
    """

    def __init__(self, p=0.5, alpha=0.1, eps=1e-6, mix='random'):
        """
        Args:
          p (float): probability of using MixStyle.
          alpha (float): parameter of the Beta distribution.
          eps (float): scaling parameter to avoid numerical issues.
          mix (str): how to mix.
        """
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps
        self.alpha = alpha
        self.mix = mix
        self._activated = True
        self.lamda= 0.8
        print(f'Using {mix} mixstyle with lambda {self.lamda}')

    def __repr__(self):
        return f'MixStyle(p={self.p}, alpha={self.alpha}, eps={self.eps}, mix={self.mix})'

    def set_activation_status(self, status=True):
        self._activated = status

    def update_mix_method(self, mix='random'):
        print(f'Updating to {mix} mixstyle')
        self.mix = mix

    # def update_mixstyle_lamda(self, lamda=None):
    #     print(f'Updating mixing coeff lamda to {lamda} mixstyle')
    #     self.lamda = lamda

    def forward(self, x):
        # print(self.training, self._activated)
        if not self.training or not self._activated:
            return x

        if random.random() > self.p:
            return x

        B = x.size(0)

        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x-mu) / sig

        lmda = self.beta.sample((B, 1, 1, 1))
        if self.lamda != None:
            lmda[:] = self.lamda
        lmda = lmda.to(x.device)

        if self.mix == 'random':
            # random shuffle
            perm = torch.randperm(B)


        mu2, sig2 = mu[perm], sig[perm]
        mu_mix = mu*lmda + mu2 * (1-lmda)
        sig_mix = sig*lmda + sig2 * (1-lmda)

        return x_normed*sig_mix + mu_mix


class MixStyle_Cls(nn.Module):
    """MixStyle.
    Reference:
      Zhou et al. Domain Generalization with MixStyle. ICLR 2021.
    """

    def __init__(self, p=0.5, alpha=0.1, eps=1e-6, mix='random', mix_sig=True):
        """
        Args:
          p (float): probability of using MixStyle.
          alpha (float): parameter of the Beta distribution.
          eps (float): scaling parameter to avoid numerical issues.
          mix (str): how to mix.
        """
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(5, 2) #torch.distributions.Beta(alpha, alpha)
        self.eps = eps
        self.alpha = 5 #alpha
        self.alpha2 = 2 #alpha
        self.mix = mix
        self._activated = True
        self.lamda= 0.9
        self.beta = torch.distributions.Beta(0.1, 0.1)
        self.mix_sig = mix_sig
        self.iter=0
        self.update_param=False
        print(f'Using {mix} mixstyle with lambda {self.lamda}, sigma mix set to {self.mix_sig}')

    def __repr__(self):
        return f'MixStyle(p={self.p}, alpha={self.alpha}, eps={self.eps}, mix={self.mix})'

    def set_activation_status(self, status=True):
        self._activated = status

    def update_mix_method(self, mix='random'):
        print(f'Updating to {mix} mixstyle')
        self.mix = mix

    def forward(self, x, labels=None):
        # print(labels, self.training, self._activated)
        if labels==None or not self.training or not self._activated:
            return x
        
        B = x.size(0)

        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x-mu) / sig

        lmda = self.beta.sample((B, 1, 1, 1))
        if self.lamda != None:
            lmda[:] = self.lamda
        lmda = lmda.to(x.device)

        if self.mix == 'sourcedomain':
            perm = torch.arange(B)
            select_srcmix = torch.randint(0, B, (3*B//4,))
            target_labels = labels

            target_mix = torch.zeros_like(select_srcmix)
            for i in range(select_srcmix.shape[0]):
                tgt_ind = torch.nonzero(target_labels==labels[select_srcmix[i]])
                n_matched = torch.numel(tgt_ind)
                if n_matched>0:
                    target_mix[i]=tgt_ind[torch.randint(0,n_matched,(1,))]
                else:
                    target_mix[i]=select_srcmix[i]
            perm[select_srcmix] = target_mix
            # print('mixing')

        elif self.mix == 'crossdomain':
            # split into two halves and swap the order
            perm = torch.arange(B - 1, -1, -1) # inverse index
            perm_b, perm_a = perm.chunk(2)
            perm_b = perm_b[torch.randperm(B // 2)]
            perm_a = perm_a[torch.randperm(B // 2)]
            perm = torch.cat([perm_b, perm_a], 0)

        else:
            raise NotImplementedError

        mu2, sig2 = mu[perm], sig[perm]
        mu_mix = mu*lmda + mu2 * (1-lmda)

        # mu_mix[B//2:] = mu[B//2:]*(1-lmda[B//2:]) + mu2[B//2:] * lmda[B//2:] #trgt mix
        if self.mix_sig:
            sig_mix = sig*lmda + sig2 * (1-lmda)
            # sig_mix[B//2:] = sig[B//2:]*(1-lmda[B//2:]) + sig2[B//2:] * lmda[B//2:] #trgt mix
        else:
            sig_mix = sig

        return x_normed*sig_mix + mu_mix