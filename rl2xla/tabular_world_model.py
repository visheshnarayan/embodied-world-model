from __future__ import annotations
import pickle
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn

class TabularDynamics(nn.Module):
    obs_dim: int; action_dim: int; latent_dim: int = 128
    @nn.compact
    def __call__(self, obs, action):
        z = nn.tanh(nn.Dense(self.latent_dim)(obs)); h = nn.tanh(nn.Dense(self.latent_dim)(jnp.concatenate([z, action], -1)))
        return nn.Dense(self.obs_dim)(nn.tanh(nn.Dense(self.latent_dim)(h)))

class TabularWorldModel:
    def __init__(self, obs_dim, action_dim, seed=0, latent_dim=128, lr=3e-4):
        self.net=TabularDynamics(obs_dim,action_dim,latent_dim); self.params=self.net.init(jax.random.PRNGKey(seed),jnp.zeros((1,obs_dim)),jnp.zeros((1,action_dim)))
        self.tx=optax.adam(lr); self.opt_state=self.tx.init(self.params); self._compiled=jax.jit(self._make_step())
    def _make_step(self):
        def step(params,opt_state,obs,action,next_obs):
            def loss_fn(p): return jnp.mean((self.net.apply(p,obs,action)-next_obs)**2)
            loss,grads=jax.value_and_grad(loss_fn)(params); updates,new=self.tx.update(grads,opt_state,params)
            return optax.apply_updates(params,updates),new,loss
        return step
    def train_step(self,obs,action,next_obs):
        self.params,self.opt_state,loss=self._compiled(self.params,self.opt_state,*map(jnp.asarray,(obs,action,next_obs))); return float(loss)
    def predict(self,obs,action): return np.asarray(self.net.apply(self.params,jnp.asarray(obs)[None],jnp.asarray(action)[None])[0])
    def predict_batch(self,obs,action): return np.asarray(self.net.apply(self.params,jnp.asarray(obs),jnp.asarray(action)))
    def save(self,path,metadata):
        with open(path,"wb") as f: pickle.dump({"params":self.params,"metadata":metadata},f)

    @classmethod
    def load(cls, path):
        with open(path,"rb") as f: state=pickle.load(f)
        meta=state["metadata"]; model=cls(meta["obs_dim"],meta["action_dim"],seed=0); model.params=state["params"]; return model,meta
