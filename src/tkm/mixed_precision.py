import jmp
from jax import jit

dot_compiled_full = jit(partial(dot,my_policy=float32))
