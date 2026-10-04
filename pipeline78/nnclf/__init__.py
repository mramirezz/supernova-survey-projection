"""Clasificador de red neuronal sobre curvas crudas g/r (GRU y transformer), exploracion frente a Villar+MCMC.

Brief: .superpowers/sdd/2026-10-02-biblioteca-a-tasas/nn-brief.md. Modulos: data (lectura, tokens, aumento,
degradacion, split por plantilla), models, train, evaluate, baseline (Villar SPM + gradient boosting) y __main__ (CLI).
Uso: python -m pipeline78.nnclf train|eval|baseline (env con torch: /opt/anaconda3/envs/series).
"""
