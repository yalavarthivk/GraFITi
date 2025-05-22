Parametrization
===============

We want to create models that can be initialized from config files.

1. Each model should have a default config, that is, up to shape information,
   sufficient to initialize the model.
2. This config should be a class that can be 1:1 translated into a ``json`` file
3. Either initialize form a config  <=> (dictionary), OR in some situations, pass a complete submodule directly.

   - This requires that the submodule has a ``from_dict`` method.

How should the recursion be performed?

Desiderata:

- all models should be fully vectorized (e.g. work with arbitrary batch sizes / number of batch dimensions.)
- all models should be able to be initialized from a config file


Usage
-----

.. code-block:: python

  class Module(bb.Module):

    @dataclass
    class Config:
       input_size: int
       output_size: int
       latent_size: int

Signatures
----------

We want to describe general signatures for models that operate in Hilbert spaces.
Generally speaking, there are 3 kinds of shape information:

1. Fixed shape sizes.
2. Variable shape sizes.
3. Outer shape sizes.

There are multiple levels at which this can be done.

1. Limited recursion: `tuple[tuple[int, ...]]`
2. Full recursion.
3. Shapes + dtypes / full specification of composite types.

   - I.e. the Hilbert space is constructed as a finite composition of sums / products of Hilbert spaces.

Examples
--------

Similarity between two images of fixed size:

- `[..., [(H, W, C), (H, W, C)]] -> ...`
