from __future__ import annotations
import pickle
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn

class GoalPolicy(nn.Module):
    action_dim: int
    @nn.compact
    def __call__(self, state, goal):
        x=jnp.concatenate([state,goal],-1); x=nn.tanh(nn.Dense(128)(x)); x=nn.tanh(nn.Dense(128)(x)); return nn.Dense(self.action_dim)(x)

class BehaviorCloningPolicy:
    def __init__(self,state_dim,action_dim,seed=0,lr=3e-4):
        self.net=GoalPolicy(action_dim); self.params=self.net.init(jax.random.PRNGKey(seed),jnp.zeros((1,state_dim)),jnp.zeros((1,state_dim))); self.tx=optax.adam(lr); self.opt_state=self.tx.init(self.params); self._compiled=jax.jit(self._make_step())
    def _make_step(self):
        def step(params,opt_state,state,goal,action):
            def loss_fn(p): return jnp.mean((self.net.apply(p,state,goal)-action)**2)
            loss,grads=jax.value_and_grad(loss_fn)(params); updates,new=self.tx.update(grads,opt_state,params); return optax.apply_updates(params,updates),new,loss
        return step
    def train_step(self,state,goal,action):
        self.params,self.opt_state,loss=self._compiled(self.params,self.opt_state,*map(jnp.asarray,(state,goal,action))); return float(loss)
    def predict(self,state,goal): return np.asarray(self.net.apply(self.params,jnp.asarray(state)[None],jnp.asarray(goal)[None])[0])
    def save(self,path,metadata):
        with open(path,"wb") as f: pickle.dump({"params":self.params,"metadata":metadata},f)
    @classmethod
    def load(cls,path):
        with open(path,"rb") as f: state=pickle.load(f)
        meta=state["metadata"]; policy=cls(meta["state_dim"],meta["action_dim"],0); policy.params=state["params"]; return policy,meta
