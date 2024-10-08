from .models import *
from .engine import *
from .datasets import *
from .evaluation import *
import mmcv
import mmengine

from mmengine.config.utils import MODULE2PACKAGE

MODULE2PACKAGE['mmsegext'] = 'mmsegext'
