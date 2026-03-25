"""Method pipelines --- one self-contained workflow per file.

Each script takes a molecular system (defaulting to H2) and produces
energy / wavefunction results using a single method variant.

Scripts (sorted by basis source)
---------------------------------
direct_ci_krylov.py
    Direct-CI (HF+S+D) basis -> Krylov time evolution.
direct_ci_sqd.py
    Direct-CI (HF+S+D) basis -> noise -> S-CORE -> batch diag.
flow_ci_krylov.py
    NF-trained + Direct-CI merged basis -> Krylov expansion.
flow_ci_sqd.py
    NF-trained + Direct-CI merged basis -> noise -> S-CORE.
flow_only_krylov.py
    NF-only basis (no CI merge) -> Krylov expansion.
flow_only_sqd.py
    NF-only basis (no CI merge) -> noise -> S-CORE.
hf_only_krylov.py
    HF-only reference state -> Krylov time evolution.
iterative_nqs_krylov.py
    Iterative NQS sampling + Krylov expansion + eigenvector feedback.
iterative_nqs_sqd.py
    Iterative NQS sampling + subspace diag + eigenvector feedback.
"""
