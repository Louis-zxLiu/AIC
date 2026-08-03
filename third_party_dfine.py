"""D-FINE 官方仓库的导入桥接层。

D-FINE 内部用相对导入且顶层包名固定为 `src`（third_party/D-FINE/src），
本项目自己的代码包已改名为 `aic` 以避免与之冲突（见 aic/ 与本文件的关系）。
这里把 third_party/D-FINE 加入 sys.path 后按其原生方式 `import src as dfine_src`，
再重新导出本项目需要的类，其余代码统一从本模块取用，不直接 `import src`。
"""

from __future__ import annotations

import os
import sys

_DFINE_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party", "D-FINE")
if _DFINE_ROOT not in sys.path:
    sys.path.insert(0, _DFINE_ROOT)

import src as dfine_src  # noqa: E402  (D-FINE 官方顶层包，故意保留原名以复用其相对导入)

from src.core import GLOBAL_CONFIG, YAMLConfig, create, register  # noqa: E402,F401
from src.nn.backbone.hgnetv2 import HGNetv2, StemBlock  # noqa: E402,F401
from src.zoo.dfine.dfine import DFINE  # noqa: E402,F401
from src.zoo.dfine.dfine_criterion import DFINECriterion  # noqa: E402,F401
from src.zoo.dfine.dfine_decoder import DFINETransformer  # noqa: E402,F401
from src.zoo.dfine.hybrid_encoder import HybridEncoder  # noqa: E402,F401
from src.zoo.dfine.matcher import HungarianMatcher  # noqa: E402,F401
from src.zoo.dfine.postprocessor import DFINEPostProcessor  # noqa: E402,F401

__all__ = [
    "dfine_src",
    "GLOBAL_CONFIG",
    "YAMLConfig",
    "create",
    "register",
    "HGNetv2",
    "StemBlock",
    "DFINE",
    "DFINECriterion",
    "DFINETransformer",
    "HybridEncoder",
    "HungarianMatcher",
    "DFINEPostProcessor",
]
