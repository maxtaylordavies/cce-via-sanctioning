import jax
import jax.numpy as jnp

N_FEATS = 3


@jax.jit
def cosine_sim(a, b):
    return jnp.dot(a, b) / (jnp.linalg.norm(a) * jnp.linalg.norm(b))


@jax.jit
def sample_plant(key, radius):
    return jax.random.uniform(key, shape=(N_FEATS,)) * radius


@jax.jit
def compute_yield(plant, recipe):
    # base energy is the magnitude of the plant vector
    base_energy = jnp.linalg.norm(plant)

    # compute cosine similarity between plant and recipe
    cos_sim = cosine_sim(plant, recipe)

    # yield is base energy scaled by cosine similarity
    return base_energy * cos_sim


plant = jnp.array([10.0, 10.0, 10.0])
print(cosine_sim(plant, jnp.array([1.0, 1.0, 1.0])))
print(cosine_sim(plant, jnp.array([2.0, 2.0, 2.0])))
print(cosine_sim(plant, jnp.array([10.0, 10.0, 10.0])))
