# locus

Head selection and subspace projection for activation steering, grounded in a property subset of the unembedding, in JAX on CPU.

pip install -e .

JAX_PLATFORMS=cpu python -m pytest

This package does not load Mistral, Llama, or DeepSeek, and it does not run the toxicity, sentiment, or sycophancy evaluations. The property dictionary is a column subset of an unembedding you pass in.
