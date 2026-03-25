Extending Samplers
==================

This guide explains how to add a new configuration sampler to qvartools.

Architecture
------------

All samplers extend the ``Sampler`` ABC defined in
``qvartools/samplers/sampler.py``:

.. code-block:: text

   Sampler (ABC)
   ├── NFSampler              (classical/)
   ├── TransformerNFSampler   (classical/)
   ├── TrotterSampler         (quantum/)
   ├── CudaQSampler           (quantum/)
   └── LUCJSampler            (quantum/)

Step 1: Implement the Sampler
------------------------------

.. code-block:: python

   # samplers/classical/my_sampler.py

   from __future__ import annotations

   import torch
   from qvartools.samplers.sampler import Sampler, SamplerResult


   class MySampler(Sampler):
       """My custom configuration sampler.

       Parameters
       ----------
       num_sites : int
           Number of lattice / orbital sites.
       temperature : float
           Sampling temperature controlling exploration.

       Examples
       --------
       >>> sampler = MySampler(num_sites=4, temperature=1.0)
       >>> result = sampler.sample(n_samples=1000)
       >>> print(f"Sampled {result.configs.shape[0]} configurations")
       """

       def __init__(
           self,
           num_sites: int,
           temperature: float = 1.0,
       ) -> None:
           self._num_sites = num_sites
           self._temperature = temperature

       def sample(self, n_samples: int) -> SamplerResult:
           """Sample configurations.

           Parameters
           ----------
           n_samples : int
               Number of configurations to sample.

           Returns
           -------
           SamplerResult
               Container with sampled configurations, counts, and metadata.
           """
           # Your sampling algorithm here
           configs = self._generate_configs(n_samples)

           # Count unique configurations
           unique, inverse, counts = torch.unique(
               configs, dim=0, return_inverse=True, return_counts=True
           )

           return SamplerResult(
               configs=unique,
               counts={
                   "".join(str(int(b)) for b in cfg): int(c)
                   for cfg, c in zip(unique, counts)
               },
               metadata={
                   "temperature": self._temperature,
                   "total_sampled": n_samples,
                   "unique_count": len(unique),
               },
           )

       def _generate_configs(self, n_samples: int) -> torch.Tensor:
           ...

Step 2: Register the Sampler
-----------------------------

Export from ``samplers/classical/__init__.py`` and ``samplers/__init__.py``.

Step 3: Add Tests
-----------------

.. code-block:: python

   # tests/test_samplers/test_my_sampler.py

   from qvartools.samplers.classical.my_sampler import MySampler

   def test_my_sampler_shape():
       sampler = MySampler(num_sites=4)
       result = sampler.sample(100)
       assert result.configs.ndim == 2
       assert result.configs.shape[1] == 4

   def test_my_sampler_valid_values():
       sampler = MySampler(num_sites=8)
       result = sampler.sample(500)
       assert (result.configs >= 0).all()
       assert (result.configs <= 1).all()

Integration with Pipelines
---------------------------

To use your sampler in a pipeline, you can either:

1. **Use it as a standalone component** by calling ``sampler.sample()`` and
   passing the configurations to a solver.

2. **Integrate with the pipeline** by modifying ``FlowGuidedKrylovPipeline``
   to accept a custom sampler in the training or basis extraction stages.

Design Guidelines
-----------------

- Return ``SamplerResult`` with ``configs`` as a 2D integer tensor
- Populate ``counts`` for reproducibility and analysis
- Use ``metadata`` for sampler-specific diagnostics
- Support a ``device`` parameter for GPU sampling
- Validate particle-number constraints when applicable
