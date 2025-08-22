from ._expand_exp import *
from ._adv_generator import *

from .boundary_expand import *
from .boundary_shrink import *
from .wood_fisher import *
from .fisher import *

from .random_label import *
from .finetune import *
from .gradient_ascent import *

from .delete import *
from .salun import *
from .bad_teacher import *


try:
    from .l2ul import *
except ModuleNotFoundError as e:
    if getattr(e, "name", "") == "advertorch":
        print("[WARN] 'advertorch' not installed; L2UL disabled.")
    else:
        raise
except ImportError as e:
    # e.g., advertorch found but incompatible (zero_gradients missing)
    print(f"[WARN] 'advertorch' incompatible ({e}); L2UL disabled.")