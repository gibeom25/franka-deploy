"""User-defined field transforms, referenced from a request schema as
"custom:user_plugins.<function_name>" (see franka_deploy/schema/plugins.py).

Each function receives a SourceContext (franka_deploy/schema/sources.py:
``.cameras`` dict, ``.robot_state`` dict) and returns whatever value that
request field needs -- this is the escape hatch for anything a generic
resize/dtype/encode pipeline can't produce, most commonly a
model-specific language-instruction embedding.

Nothing here is imported by default; this file only runs if a schema
actually references one of its functions.
"""

from __future__ import annotations

import numpy as np


def embed_instruction(ctx) -> list:
    """Placeholder -- replace with your model's actual text encoder.

    Referenced by configs/examples/libero_style_example.yaml's "lang_emb"
    field (float32 [1, 512]). Loading a real encoder here (e.g. a small
    local CLIP/T5 text tower) is the user's responsibility -- franka-deploy
    has no way to know which one a given policy server was trained against.
    """
    return np.zeros((1, 512), dtype=np.float32).tolist()
