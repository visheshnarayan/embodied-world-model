from __future__ import annotations
import pickle
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn

class LatentDynamics(nn.Module):
    latent_dim:int=32
    @nn.compact
    def __call__(self,obs,action):
        z=nn.tanh(nn.Dense(self.latent_dim)(obs)); h=nn.tanh(nn.Dense(64)(jnp.concatenate([z,action],-1)))
        next_obs=nn.tanh(nn.Dense(8)(h)); return next_obs,nn.Dense(1)(h)[...,0]

class WorldModel:
    def __init__(self,seed=0,latent_dim=32,lr=3e-4):
        self.net=LatentDynamics(latent_dim); self.params=self.net.init(jax.random.PRNGKey(seed),jnp.zeros((1,8)),jnp.zeros((1,2)))
        self.tx=optax.adam(lr); self.opt_state=self.tx.init(self.params); self.lr=lr
        self._compiled=jax.jit(self._make_step())
        self._predict_compiled=jax.jit(lambda params, obs, action: self.net.apply(params, obs, action))
    def _loss(self,p,obs,action,reward,next_obs):
        pred,r=self.net.apply(p,obs,action)
        return jnp.mean((pred-next_obs)**2)+.1*jnp.mean((r-reward)**2)
    def _make_step(self):
        def step(params,opt_state,obs,action,reward,next_obs):
            loss,grads=jax.value_and_grad(self._loss)(params,obs,action,reward,next_obs); updates,new=self.tx.update(grads,opt_state,params)
            return optax.apply_updates(params,updates),new,loss
        return step
    def train_step(self,obs,action,reward,next_obs):
        self.params,self.opt_state,loss=self._compiled(self.params,self.opt_state,*map(jnp.asarray,(obs,action,reward,next_obs))); return float(loss)
    def predict(self,obs,action):
        z,r=self._predict_compiled(self.params,jnp.asarray(obs)[None],jnp.asarray(action)[None]); return np.asarray(z[0]),float(r[0])
    def predict_batch(self,obs,action):
        """Predict a batch of imagined transitions in one JAX call."""
        z,r=self._predict_compiled(self.params,jnp.asarray(obs),jnp.asarray(action))
        return np.asarray(z),np.asarray(r)
    def save(self,path):
        with open(path,"wb") as f: pickle.dump(self.params,f)
    def load(self,path):
        with open(path,"rb") as f: self.params=pickle.load(f)
