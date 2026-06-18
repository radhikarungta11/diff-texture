from __future__ import annotations

from src.config import CFG
from src.optimize_texture import TextureOptimizer


if __name__ == '__main__':
    optimizer = TextureOptimizer(CFG)
    optimizer.run()