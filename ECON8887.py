#!/usr/bin/env python
# coding: utf-8

# In[83]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import scipy as sp
from scipy.stats import norm, lognorm


# # HUMAN CAPITAL, GROWTH AND INEQUALITY

# ## Notation
# 
# * $t \in \mathbb{N} := \{0,1,2, ...\}$ : Date / time 
# 
# * $c_{t}$ : Consumption outcome of a $y$oung agent in date $t$
# 
# * $d_{t}$ : Consumption outcome of an $o$ld agent in date $t$
# 
# * $s_{t}$ : Savings by a young agent in date $t$
# 
# * $w_{t}$ : Real wage rate (in units of date $t$ consumption)
# 
# * $R_{t+1}$ : Relative price of date-$(t+1)$ consumption (in units of date $t$ consumption) 
# 
# * $k_{t}$, ($y_{t}$) : Per-worker capital stock (output) in terms of "efficient" worker units
# 
# * $e_{t}$ : Education expenditure
# 
# * $h_{t}$ : Individual's stock of human capital
# 
# * $H_{t}$ : Aggregate stock of human capital
# 
# * $l_{t}$ : $\ln(h_{t})$ 
# 
# * $1$ : Normalized individual time endowment

# ## Human capital evolution
# 
# Assumptions:
# 
# * It takes time for current educational investment $e_{t}$ to be embodied in next-period children. 
# 
# * A child's human capital ($h_{t+1}$) will also inherit from her parents' human capital ($h_{t}$). 
# 
# Evolution (or production) of within-household human capital</a>:
# 
# \begin{equation}
# 	h_{t+1} =  Ae_{t}^{\theta}h_{t}^{1-\theta}, \qquad \theta \in (0,1), A > 0.
# 	\label{eq:OLG human capital production}
# \end{equation}

# ## Household (agent)
# 
# Lifetime utility:
# 
# \begin{equation}
# 	U(c_t) + \beta U(d_{t+1}) + \gamma U(e_{t})
# 	\label{eq: OLG lifetime utility}
# \end{equation}
# 
# 
# * $\beta \in (0,1)$ (household impatience)
# 
# * $\gamma > 0$ (degree of altruism towards one's own children)
# 
# Assume $U(x) = \ln (x)$.
# 
# 
# The date $t$ household faces these budget constraints over their lifetime:
# \begin{equation}
# 	c_{t} + s_{t} + e_{t} = w_t h_{t},
# 	\label{eq: OLG young BC}
# \end{equation}
# and,
# \begin{equation}
# 	d_{t+1}  = R_{t+1}s_t
# 	\label{eq: OLG old BC}
# \end{equation}

# ## Firm
# 
# A representative firm produces a per-person final good using the technology:
# 
# \begin{equation}
# 	f(k_{t}) := Z k_{t}^{\alpha}, \qquad \alpha \in (0,1), Z >0.
# 	\label{eq: OLG final goods production}
# \end{equation}
# 
# The variable $k_{t} := K_{t}/H_{t}$ is the aggregate capital-to-efficient-labor ratio.
# 
# Assume capital depreciates fully each period.
# 
# Profit maximization gives the firm's optimal demand for effective labor and capital</a>, respectively, as
# 
# \begin{equation}
# 	w_{t} =  (1-\alpha)Zk_{t}^{\alpha},
# 	\label{eq: OLG labor demand}
# \end{equation}
# 
# and,
# 
# \begin{equation}
# 	R_{t} =  \alpha Zk_{t}^{\alpha - 1}.
# 	\label{eq: OLG capital demand}
# \end{equation}
# 
# 

# ## Aggregation and market clearing
# 
# * Let the cumulative probability distribution function over $h_{t}$ at date $t$ be $M_{t}: \mathbb{R}_{++} \mapsto [0,1]$. 
# 
#     * The function $M_{t}$ is non-decreasing, 
#     
#     * $M_{t}(0^{+}) = 0$ and $M_{t}(+\infty) = 1$, where $0^{+} := \lim_{h \searrow 0} h$.  
#     
# * Let the distribution of the logarithm of $h_{t}$ be given by the c.d.f. $\mu_{t}$. Note:
# 
#     $$
#     \mu_{t}(\ln (h_{t})) = M_{t}(h_{t}).
#     $$
# 
# * Assume initially, $\mu_{0}$, has variance $\sigma_0 > 0$.

# Thus, we can account for the aggregate level of human capital through labor market clearing</a>:
# $$
# 	H_{t} := \int \underbrace{e^{\ln (h_{t})}}_{\text{Individual } h_{t}} \text{d}\mu_{t}(\ln (h_{t})).
# 	\label{eq: OLG aggregate H capital}
# $$
# 
# Capital market clearing</a> is summarized by
# \begin{equation}
# 	\underbrace{K_{t+1}}_{\text{Aggregate capital for next period}} = \int \underbrace{e^{\ln (s_{t})}}_{\text{Individual } s_{t}} \text{d}\mu_{t}(\ln (h_{t})).
# 	\label{eq: OLG capital clearing}
# \end{equation} 
# 

# Plug optimal education demand into the evolution of individual human capital, we get:
# 
# $$
# \frac{h_{t+1}}{h_{t}} = A\left[\frac{\gamma}{1+\beta+\gamma}w_{t}\right]^{\theta}.
# $$
# 
# Growth rate in *individual household's* human capital</a> is a positive-valued function of the aggregate outcome.

# Now, dividing through by $H_{t+1}$ on both sides,
# 
# $$
# \frac{K_{t+1}}{H_{t+1}} = \frac{\beta}{1+\beta +\gamma}w_{t}\frac{H_{t}}{H_{t+1}}
# $$
# 
# Using the result on the growth rate of aggregate human capital, we can write
# 
# $$
# \frac{K_{t+1}}{H_{t+1}} = \frac{\beta \kappa^{-1}}{1+\beta +\gamma}w_{t}^{1-\theta}, \qquad \kappa := A\left(\frac{\gamma}{1+\beta+\gamma}\right)^{\theta}.
# $$

# ## Consequence for distribution of wealth/income

# Let $\bar{l}_{t+1} = \mathbb{E}_{t}\ln (h_{t+1})$ and $\bar{l}_{t} = \mathbb{E}_{t}\ln (h_{t})$ (expectations conditional on $w_{t}$).
# 
# Taking conditional expectations of the evolution of log human capital, the mean of the distribution of human capital</a> follows:
# 
# $$
# \bar{l}_{t+1} = \bar{l}_{t} + v_{t}, \qquad v_{t} := \ln (\kappa) + \theta \ln(w_{t}),
# $$
# 			

# Remarks:
#     
# * For the model to be interesting, we would like to study cases where $\bar{l}_{t+1} - \bar{l}_{t} = \nu_{t} > 0$.
# 
# * That is human capital growth rate is positive.
# 
# * A sufficient condition on parameters is: 
# 
#     $$
#     \kappa w_{0}^{\theta} > 1 \qquad \Longleftarrow \qquad A > A_{min} := \left(\frac{\gamma w_0}{1+\beta+\gamma}\right)^{-\theta}
#     $$
#     
#     where $w_0 = (1-\alpha)(k_0)^{\alpha}$ is the initial RCE wage rate, given initial $k_0$.

# The evolution of the distributions' variances</a> follows:
# 
# $$
# \mathbb{E}_{t}(\ln (h_{t+1}) - \bar{l}_{t+1})^{2} = \mathbb{E}_{t}\left[ (\ln (h_{t}) + \nu_{t}) - (\bar{l}_{t} + \nu_{t})\right]^{2}
# $$
# 
# Or we can denote this as:
# 
# $$
# \sigma_{t+1}^{2} = \sigma_{t}^{2} = \sigma_{0}^{2}.
# $$

# In[84]:


class Inequality:
    
    
    def __init__(self,
                           α = 0.33,
                           β = 1.0/(1.0+0.04**35.),
                           γ = 0.8,
                           θ = 0.25,
                           A = 1.0,
                           Z = 1.0,
                           k0 = 0.01):
        
        self.α, self.β, self.γ, self.θ, self.A, self.Z, self.k0 = α, β, γ, θ, A, Z, k0
        self.w0 = (1.0-self.α)*self.k0**self.α
               
        A_min = (self.γ*self.w0/(1+self.β+self.γ))**(-self.θ)


        if self.A <= A_min:
            self.A = A_min + 0.01
            print("This is to ensure positive growth in human-capital accumulation")
        
        self.κ = self.A*(self.γ/(1+self.β+self.γ))**self.θ  
        
    
    def f(self, k):
        # output per efficiency unit of workers
        y = self.Z*k**self.α   
        # Associated equilibrium relative price
        R = self.α*self.Z*k**(self.α-1.0)
        w = (1.0-self.α)*self.Z*k**self.α
        
        return y, R, w
    
    def g_prive(self, k):
        
        knext = ((self.β/self.κ)/(1+self.β+self.γ))*((1.0-self.α)*self.Z*(k**self.α))**(1.0-self.θ)
        
        return knext
    
    def moment_update(self, k, lbar, sigma2):
        """Recursion on Normal distribution for log-h - mean and variance"""

        # Mean and variance of Normal dist. of log(h)
        lbar_next = lbar + np.log(self.κ) + self.θ*np.log(self.f(k)[2])
        sigma2_next = sigma2

        return lbar_next, sigma2_next


    def distro_log2level(self, k, lbar, sigma2):
        """Transform unique Normal distribution of log-h indexed by moments (lbar, sigma2) into implied log-Normal distribution M of level of h.
           Also calculate distribution of auxiliary variables: labor income, capital income and wealth."""
    
        mean_h = np.exp(lbar + sigma2/2.0)
        var_h = np.exp(2.0*lbar + sigma2)*(np.exp(sigma2) - 1.0)
        
        dist = lognorm(s=np.sqrt(var_h), loc=lbar, scale=mean_h) 

        return out
            
        
    


# In[98]:


k0 = 0.01
lbar0 = 0.0
In = Inequality()

def plot_normal(k, lbar, sigma2=0.25, T=10):
        """Simulate RCE outcomes given initial conditions"""
        fig, ax = plt.subplots() 
        
        for t in range(T):
            dist = norm(loc=lbar, scale=np.sqrt(sigma2))
            
            lbar_next, sigma2_next = In.moment_update(k, lbar, sigma2)
   
            k = In.g_prive(k)
            lbar = lbar_next
            sigma2 = sigma2_next
            
            
            a, b = dist.interval(0.99999)
            x=np.linspace(a, b, 100)
            plt.plot(x, dist.pdf(x), label=str("$t$ = ") + str(t))
            plt.xlabel("Individual log human capital, $\ln (h_{t})$")
        plt.legend()    
        plt.show()


# In[99]:


plot_normal(k0, lbar0)


# In[100]:


def plot_lognormal(k, lbar, sigma2=0.25, T=10):
        """Simulate RCE outcomes given initial conditions"""
        fig, ax = plt.subplots()
        
        for t in range(T):
            mean_h = np.exp(lbar)
            dist = lognorm(s=np.sqrt(sigma2), scale=mean_h) 
            
            lbar_next, sigma2_next = In.moment_update(k, lbar, sigma2)
   
            k = In.g_prive(k)
            lbar = lbar_next
            sigma2 = sigma2_next
            
            
            a, b = dist.interval(0.99999)
            x=np.linspace(a, b, 100)
            plt.plot(x, dist.pdf(x), label=str("$t$ = ") + str(t))
            plt.xlabel("Individual human capital, $h_{t}$")
        plt.legend()    
        plt.show()    
            


# In[101]:


plot_lognormal(k0, lbar0)


# In[102]:


def plot_lognormal_wealth(k, lbar, sigma2=0.25, T=10):
        """Simulate RCE outcomes given initial conditions"""
        fig, ax = plt.subplots()
        
        for t in range(T):
            constant = np.log(In.β/(1+In.β+In.γ))
            lw = np.log(In.f(k)[2])
            mean_wealth = np.exp(lbar+lw+constant)
            dist = lognorm(s=np.sqrt(sigma2), scale=mean_wealth)
            
            lbar_next, sigma2_next = In.moment_update(k, lbar, sigma2)
   
            k = In.g_prive(k)
            lbar = lbar_next
            sigma2 = sigma2_next
            
            
            a, b = dist.interval(0.99999)
            x=np.linspace(a, b, 100)
            plt.plot(x, dist.pdf(x), label=str("$t$ = ") + str(t))
            plt.xlabel("Individual wealth, $s_{t}$")
            
        plt.legend()    
        plt.show()    


# In[103]:


plot_lognormal_wealth(k0, lbar0)


# In[ ]:




