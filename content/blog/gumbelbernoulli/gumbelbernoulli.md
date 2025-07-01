---
title: The Gumbel-max trick for the Bernoulli distribution
description: This blogpost described the Gumbel-max trick for the Bernoulli distribution
date: 2025-07-01
tags:
  - research
---

In deep learning, the Gumbel-Max Trick is used to sample from a categorical distribution during a forward pass, while keeping the whole differentiable.
It is useful in situations where discrete values should be obtained as some intermediate representation.
In such cases, the Gumbel-Max trick enables differentiable sampling (or gradients to flow through sampling operations).
In this sense, it is similar to the reparameterization trick used in VAEs, but then for categorical distributions.
It is popular to use in, for example, discrete VAEs[^dvae].
I will not fully reintroduce the concepts and maths behind the Gumbel-Max trick here.
Instead, refer to [this blog](https://sassafras13.github.io/GumbelSoftmax/) for a good introduction.

This blogpost will focus on adapting the Gumbel-max trick for the Bernoulli distribution.
Usually, the Gumbel-max trick is paired with the softmax to reparameterize a $k$-class categorical distribution.
For example, the bottleneck space of a discrete autoencoder could contain a categorical "concept" per patch of locally aggregated pixels.
Other cases might exist where the internals should be either 0 or 1.
Take, for instance, a case where the discretized bottleneck space should encode whether a certain concept is present or not[^note].

In neural networks, binary (i.e. Bernoulli) random variables are often modeled with the sigmoid function $\sigma$:
$$p = \sigma(x) = \frac{1}{1+e^{-x}} = \frac{e^x}{e^x+1},$$
and
$$1-p = 1 - \sigma(x) = 1 - \frac{e^x}{e^x+1} = \frac{e^x+1-e^x}{e^x+1} = \frac{1}{e^x+1}.$$

Both quantities would represent the estimated probability ($p$) of an event occurring, or not, respectively.

The sigmoid operation can be rewritten as a special case of the two-way softmax.
Using 2 unnormalized log-probabilities (i.e. logits) $l_1$ and $l_2$, the two-way softmax can compute the same probabilities:
$$p = \frac{e^{l_1}}{e^{l_1}+e^{l_2}},$$
and
$$1-p = \frac{e^{l_2}}{e^{l_1}+e^{l_2}}.$$

These equations allow us to trivially see that the sigmoid function is equivalent to a two-way softmax where the two (unnormalized log-probabilites) are $l_1=x$ and $l_2=0$.
As such, we can apply the Gumbel-Max trick as one does with the softmax operation, and afterwards rewrite it to logistic form.

The Gumbel-Max trick simply involves adding Gumbel-noise to the logits.
Let us first sample two points ($g_1$ and $g_2$) from the Gumbel distribution:
$$g_1, g_2 \sim \mathrm{Gumbel}(0,1)$$

Then, we add these to our logits $l_1=x$ and $l_2=0$ in the sigmoid operation:
$$\text{Gumbel - }\sigma(x) = \frac{e^{x+g_1}}{e^{x+g_1}+e^{g_2}} = \frac{1}{(e^{x+g_1}+e^{g_2})e^{-(x+g_1)}} = \frac{1}{e^{-(x+g_1-g_2)}} = \sigma(x+g_1-g_2).$$
As such, applying the reparameterization Gumbel-Max trick for Bernoulli random variables (parameterized using the sigmoid operation) involves adding the difference of two Gumbels.
This has been described before by Maddison *et al.*[^concrete].

The python code for the Gumbel-sigmoid operation is:
```python
def gumbel_sigmoid(logits, tau=1, hard=False):
    gumbels_1 = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format)
        .exponential_()
        .log()
    )
    gumbels_2 = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format)
        .exponential_()
        .log()
    )

    y_soft = torch.sigmoid((logits + gumbels_1 - gumbels_2) / tau)

    if hard:
        indices = (y_soft > .5).nonzero(as_tuple=True)
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format)
        y_hard[indices[0], indices[1]] = 1.0
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    return ret
```




[^dvae]: Jang, Eric, Shixiang Gu, and Ben Poole. "Categorical reparameterization with gumbel-softmax." arXiv preprint arXiv:1611.01144 (2016).

[^concrete]: Maddison, Chris J., Andriy Mnih, and Yee Whye Teh. "The concrete distribution: A continuous relaxation of discrete random variables." arXiv preprint arXiv:1611.00712 (2016).

[^note]: Of course, it is also always possible to output 2 neurons, pairing it with the usual Gumbel-softmax operation.