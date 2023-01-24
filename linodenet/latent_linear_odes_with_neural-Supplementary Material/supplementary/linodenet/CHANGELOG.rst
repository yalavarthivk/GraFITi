CHANGELOG
=========

2021-10-18 version 0.1.6
------------------------

- Moved Documentation to Sphinx-AutoAPI
- Added ``register_buffer`` to most models
    - :class:`linodenet.model.LinearContraction` tracks ``spectral_norm``
    - :class:`linodenet.model.LinODEnet` tracks ``xhat_pre``, ``xhat_post``, ``zhat_pre``, ``zhat_post``,
      but only outputs ``xhat_post`` now.
- Improved performance of pipeline by >100% through the use of more sensible cache policies
- Namespace are now cleaner (e.g. ``linodenet.models.iResNet`` instead of ``linodenet.models._iresnet.iResNet``)
- added ``functional`` and ``modular`` distinction for ``initializations``, ``projections`` and ``regularizations``

2021-09-27 version 0.1.5
------------------------
- Changed usage of torchscript to non-deprecated version.
    - Added :func:`linodenet.util.autojit` decorator to automatically jit-compile classes upon instantiation.
- fixed order ``xhat_post`` and ``xhat_pre``
- added config module
