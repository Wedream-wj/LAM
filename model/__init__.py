# -*- coding: utf-8 -*-
# @Time    : 7/15/2021 3:52 PM
# @Author  : YaoGengqi
# @FileName: __init__.py
# @Software: PyCharm
# @Description:

from .IMDN import get_IMDN
from .block import VGGFeatureExtractor as get_Extractor
from .EdgeSRN import get_EdgeSRN

def get_model(model_name, checkpoint, upscale):

    if model_name == 'IMDN':
        return get_IMDN(upscale=upscale, checkpoint=checkpoint)

    elif model_name == 'EdgeSRN':
        return get_EdgeSRN(checkpoint=checkpoint)

