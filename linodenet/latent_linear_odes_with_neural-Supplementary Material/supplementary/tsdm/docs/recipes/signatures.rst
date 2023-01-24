Signatures for Functions between Hilbert Spaces
===============================================

We consider the universe of Hilbert spaces

.. math:: 𝓤 = (ℝ, ⊕, ⊗, *)

which consists of all Hilbert spaces that cen be constructed by a finite number of direct sums and inner products,
and dual operations from the set of real numbers. In particular, $𝓤$ contains all Hilbert spaces of the form $ℝⁿ$ for
some $n$ and also the important infinite dimensional Hilbert space $ℝ$.

Finally, we also want to be able to discuss **sequence-space**. Since the computer can only consider finite sequences,
we consider the space of eventually zero-terminated sequences. Or rather, the union $⋃_{n=0}^∞ℝ^n$.

We ask what is a way to represent signatures of functions between these Hilbert spaces.
In particular, we want to specify a mini-language that is compatible with previous attempts.

Representing simple tensors
---------------------------

To represent a simple tensor such as $ℝ^m⊗ℝ^n$ in $𝓤$, we use a tuple of integers. `(m, n)`.

- To represent the elementary vector spaces, we use integers representing their dimension, e.g `2` for $ℝ^2$.

  - We allow ourselves to use variables, i.e. `m` for $ℝ^m$, if the size is unknown a-priori.
  - We use negative numbers to represent dual spaces, i.e. `-2` for $(ℝ^2)^*$.

- To represent the space of zero-terminated sequences, we use ???
- To represent direct sums, we use lists, e.g. `[m, n]` represents $ℝ^m⊕ℝ^n$.
- To represent tensor products, we use tuples, e.g. `(m, n)` represents $ℝ^m⊗ℝ^n$.
- To represent batching, we use the `Ellipsis` object, e.g. `[..., m, n]` represents any object of the form $V⊗ℝ^m⊗ℝ^n$,
  in particular $ℝ^{?}⊗ℝ^m⊗ℝ^n$,

  -  Batching is only allowed once, i.e. all occurrences of `Ellipsis` must refer to the same variable.



Note that we consider the addition of the dual operator as optional.
We model it by using negative numbers in the exponents.


Variable Dimensional Inputs
---------------------------

We distinguish a few important cases:

- Inputs of known dimensionality

  - The dimensionality of the input is known at compile-time and known at runtime.
  - Example: $ℝ^3→ℝ^3$, `(3,) -> (3,)`

- Inputs of a-priori unknown dimensionality

  - The dimensionality of the input is unknown at compile-time and known at runtime.
  - Example: Einsum operator: $(A, x) ↦ A⋅x$, `[(m, n), (n,)] ↦ (m,)`, `ij, j-> i`

- A-posteriori variable dimensional inputs

  - The dimensionality of the input is unknown at compile-time and unknown at runtime.
  - Example: Mapping a sequence to a scalar: $(s_n)_{n=1:N} ↦ c$

Representing Functions between Vector Spaces
--------------------------------------------



Dimension counting
------------------

To get the dimension of a tensor from a signature, we simply need to:

- sum up the absolute values across direct sums
- multiply the absolute values across tensor products

If we want to include the signs, note that due to commutativity of direct sums, tensor products and the distributivity
and self-inversion properties of the dual operator, we have that:

- Any direct sum :math:`⨁(ℝ^{n_k})^{a_k}`, where :math:`a_k∈\{1, *\}`, can be expressed equivalently as

  .. math:: \Big(⨁_{k: a_k=1} ℝ^{n_k}\Big) ⊕ \Big(⨁_{k: a_k=*} ℝ^{n_k}\Big)^*

- Any tensor product :math:`⨂ (ℝ^{n_k})^{a_k}`, where :math:`a_k∈\{1, *\}`, can be expressed equivalently as

  .. math:: \Big(⨂_{k: a_k=1} ℝ^{n_k}\Big) ⨂ \Big(⨂_{k: a_k=*} ℝ^{n_k}\Big)^*


If we sum up the values with their signs, we get the dimension change the tensor introduces, when we consider it as a
linear map applied to some tensor equal to the dual part of the signature.

Examples:

- A matrix vector product $(A,x)↦A⋅x$ is equivalent to the signature `[(m,-n), (n,)] -> (m,)`


Einstein Summations
-------------------

The notation introduced can be used to perform Einstein summations.

Things to implement
-------------------

- Variable Class

  - Based on `sympy.Symbols`?
  - By default, variables are assumed to be real-valued, but this can be changed.
  - Allow Variables to be data-types. (int32, float32, complex64, etc.)

- Signature Class

  - Recursive definition with `[]` and `()` as containers and `(int, Ellipsis, str, sp.Symbol)` as elements

- Signature Parser (string -> signature)
- Signature Representation (signature -> string)
- Signature Chaining (signature, signature -> signature)


Problems
--------

The space of zero-terminated sequence is not a Hilbert space. Instead, we should somehow consider
the union  $⋃_{n=0}^∞ℝ^n$, together with universal linear transformations. (such as sum, mean, scalar multiplication,
trace, diag, etc.)
