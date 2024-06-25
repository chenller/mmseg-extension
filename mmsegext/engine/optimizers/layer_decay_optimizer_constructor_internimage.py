# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Mostly copy-paste from BEiT library:

https://github.com/microsoft/unilm/blob/master/beit/semantic_segmentation/mmcv_custom/layer_decay_optimizer_constructor.py
"""

import json
import logging
from mmengine.dist import get_dist_info
from mmseg.registry import OPTIM_WRAPPER_CONSTRUCTORS
from mmengine.optim import DefaultOptimWrapperConstructor
import torch.distributed as dist

logger_initialized = {}


def get_logger(name, log_file=None, log_level=logging.INFO, file_mode='w'):
    """Initialize and get a logger by name.

    If the logger has not been initialized, this method will initialize the
    logger by adding one or two handlers, otherwise the initialized logger will
    be directly returned. During initialization, a StreamHandler will always be
    added. If `log_file` is specified and the process rank is 0, a FileHandler
    will also be added.

    Args:
        name (str): Logger name.
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the logger.
        log_level (int): The logger level. Note that only the process of
            rank 0 is affected, and other processes will set the level to
            "Error" thus be silent most of the time.
        file_mode (str): The file mode used in opening log file.
            Defaults to 'w'.

    Returns:
        logging.Logger: The expected logger.
    """
    logger = logging.getLogger(name)
    if name in logger_initialized:
        return logger
    # handle hierarchical names
    # e.g., logger "a" is initialized, then logger "a.b" will skip the
    # initialization since it is a child of "a".
    for logger_name in logger_initialized:
        if name.startswith(logger_name):
            return logger

    # handle duplicate logs to the console
    # Starting in 1.8.0, PyTorch DDP attaches a StreamHandler <stderr> (NOTSET)
    # to the root logger. As logger.propagate is True by default, this root
    # level handler causes logging messages from rank>0 processes to
    # unexpectedly show up on the console, creating much unwanted clutter.
    # To fix this issue, we set the root logger's StreamHandler, if any, to log
    # at the ERROR level.
    for handler in logger.root.handlers:
        if type(handler) is logging.StreamHandler:
            handler.setLevel(logging.ERROR)

    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    # only rank 0 will add a FileHandler
    if rank == 0 and log_file is not None:
        # Here, the default behaviour of the official logger is 'a'. Thus, we
        # provide an interface to change the file mode to the default
        # behaviour.
        file_handler = logging.FileHandler(log_file, file_mode)
        handlers.append(file_handler)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    if rank == 0:
        logger.setLevel(log_level)
    else:
        logger.setLevel(logging.ERROR)

    logger_initialized[name] = True

    return logger


def get_num_layer_for_swin(var_name, num_max_layer, depths):
    if var_name.startswith("backbone.patch_embed"):
        return 0
    elif var_name.startswith('decode_head.mask_embed'):
        return 0
    elif var_name.startswith('decode_head.cls_embed'):
        return 0
    elif var_name.startswith('decode_head.level_embed'):
        return 0
    elif var_name.startswith('decode_head.query_embed'):
        return 0
    elif var_name.startswith('decode_head.query_feat'):
        return 0
    if var_name.startswith("backbone.cb_modules.0.patch_embed"):
        return 0
    elif "level_embeds" in var_name:
        return 0
    elif var_name.startswith("backbone.layers") or var_name.startswith(
            "backbone.levels"):
        if var_name.split('.')[3] not in ['downsample', 'norm']:
            stage_id = int(var_name.split('.')[2])
            layer_id = int(var_name.split('.')[4])
            # layers for Swin-Large: [2, 2, 18, 2]
            if stage_id == 0:
                return layer_id + 1
            elif stage_id == 1:
                return layer_id + 1 + depths[0]
            elif stage_id == 2:
                return layer_id + 1 + depths[0] + depths[1]
            else:
                return layer_id + 1 + depths[0] + depths[1] + depths[2]
        else:
            stage_id = int(var_name.split('.')[2])
            if stage_id == 0:
                return 1 + depths[0]
            elif stage_id == 1:
                return 1 + depths[0] + depths[1]
            elif stage_id == 2:
                return 1 + depths[0] + depths[1] + depths[2]
            else:
                return 1 + depths[0] + depths[1] + depths[2]
    else:
        return num_max_layer - 1


@OPTIM_WRAPPER_CONSTRUCTORS.register_module(name='ext-LayerDecayOptimizerConstructor-InternImage')
class LayerDecayOptimizerConstructorInternImage(DefaultOptimWrapperConstructor):

    def add_params(self, params, module, prefix='', is_dcn_module=None):
        """Add all parameters of module to the params list.
        The parameters of the given module will be added to the list of param
        groups, with specific rules defined by paramwise_cfg.
        Args:
            params (list[dict]): A list of param groups, it will be modified
                in place.
            module (nn.Module): The module to be added.
            prefix (str): The prefix of the module
            is_dcn_module (int|float|None): If the current module is a
                submodule of DCN, `is_dcn_module` will be passed to
                control conv_offset layer's learning rate. Defaults to None.
        """
        parameter_groups = {}
        logger = get_logger(name='mmseg')
        logger.info(self.paramwise_cfg)
        backbone_small_lr = self.paramwise_cfg.get('backbone_small_lr', False)
        dino_head = self.paramwise_cfg.get('dino_head', False)
        num_layers = self.paramwise_cfg.get('num_layers') + 2
        layer_decay_rate = self.paramwise_cfg.get('layer_decay_rate')
        depths = self.paramwise_cfg.get('depths')
        offset_lr_scale = self.paramwise_cfg.get('offset_lr_scale', 1.0)

        logger.info("Build CustomLayerDecayOptimizerConstructor %f - %d" %
                    (layer_decay_rate, num_layers))
        weight_decay = self.base_wd

        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if len(param.shape) == 1 or name.endswith(".bias") or \
                    "relative_position" in name or \
                    "norm" in name or \
                    "sampling_offsets" in name:
                group_name = "no_decay"
                this_weight_decay = 0.
            else:
                group_name = "decay"
                this_weight_decay = weight_decay

            layer_id = get_num_layer_for_swin(name, num_layers, depths)
            if layer_id == num_layers - 1 and dino_head and \
                    ("sampling_offsets" in name or "reference_points" in name):
                group_name = "layer_%d_%s_0.1x" % (layer_id, group_name)
            elif ("sampling_offsets" in name or "reference_points" in name) and "backbone" in name:
                group_name = "layer_%d_%s_offset_lr_scale" % (layer_id,
                                                              group_name)
            else:
                group_name = "layer_%d_%s" % (layer_id, group_name)

            if group_name not in parameter_groups:
                scale = layer_decay_rate ** (num_layers - layer_id - 1)
                if scale < 1 and backbone_small_lr == True:
                    scale = scale * 0.1
                if "0.1x" in group_name:
                    scale = scale * 0.1
                if "offset_lr_scale" in group_name:
                    scale = scale * offset_lr_scale

                parameter_groups[group_name] = {
                    "weight_decay": this_weight_decay,
                    "params": [],
                    "param_names": [],
                    "lr_scale": scale,
                    "group_name": group_name,
                    "lr": scale * self.base_lr,
                }

            parameter_groups[group_name]["params"].append(param)
            parameter_groups[group_name]["param_names"].append(name)
        rank, _ = get_dist_info()
        if rank == 0:
            to_display = {}
            for key in parameter_groups:
                to_display[key] = {
                    "param_names": parameter_groups[key]["param_names"],
                    "lr_scale": parameter_groups[key]["lr_scale"],
                    "lr": parameter_groups[key]["lr"],
                    "weight_decay": parameter_groups[key]["weight_decay"],
                }
            logger.info("Param groups = %s" % json.dumps(to_display, indent=2))

        # state_dict = module.state_dict()
        # for group_name in parameter_groups:
        #     group = parameter_groups[group_name]
        #     for name in group["param_names"]:
        #         group["params"].append(state_dict[name])

        params.extend(parameter_groups.values())
