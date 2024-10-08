import logging
from collections import OrderedDict
from typing import Optional, Dict, Sequence, Union

import numpy as np
from mmengine import print_log, MMLogger
from mmengine.dist import is_main_process, collect_results, broadcast_object_list
from mmengine.evaluator import BaseMetric
from mmengine.evaluator.metric import _to_cpu
from prettytable import PrettyTable

from .miou_dic_utils import AccuracyMetricList
from mmseg.registry import METRICS


@METRICS.register_module(name='ext-IoUDICMetric')
class IoUDICMetric(BaseMetric):
    ioufun: AccuracyMetricList = None

    def __init__(self, accuracyD: bool = True,
                 accuracyI: bool = True,
                 accuracyC: bool = True,
                 q: int = 1,
                 binary: bool = False,
                 ignore_index: int = 255,  # end

                 format_only: bool = False,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 **kwargs):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.accuracyD = accuracyD
        self.accuracyI = accuracyI
        self.accuracyC = accuracyC
        self.q = q
        self.binary = binary
        self.ignore_index = ignore_index
        self.format_only = format_only

        self.logger: MMLogger = MMLogger.get_current_instance()

    def print_log(self, msg, level=logging.INFO):
        print_log(f"{self.__class__.__name__} --> {msg}", logger=self.logger, level=level)

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data and data_samples.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        num_classes = len(self.dataset_meta['classes'])
        if self.ioufun is None:
            self.ioufun = AccuracyMetricList(accuracyD=self.accuracyD, accuracyI=self.accuracyI,
                                             accuracyC=self.accuracyC,
                                             q=self.q, binary=self.binary, num_classes=num_classes,
                                             ignore_index=self.ignore_index)
        for data_sample in data_samples:
            pred_label = data_sample['pred_sem_seg']['data']
            # format_only always for test dataset without ground truth
            if not self.format_only:
                label = data_sample['gt_sem_seg']['data'].to(pred_label)
                self.ioufun.add(pred=pred_label, label=label)

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
                the metrics, and the values are corresponding results. The key
                mainly includes aAcc, mIoU, mAcc, mDice, mFscore, mPrecision,
                mRecall.
                example:
                {'Acc': tensor(10.6152), 'mAccD': tensor(10.6091), 'mIoUD': tensor(5.3075), 'mDiceD': tensor(10.0355),
                 'mAccI': tensor(10.6206), 'mIoUI': tensor(6.3304), 'mDiceI': tensor(11.8913), 'mAccIQ': tensor(10.2599),
                 'mIoUIQ': tensor(5.8092), 'mDiceIQ': tensor(10.9739), 'mAccI1': tensor(10.0889), 'mIoUI1': tensor(5.5769),
                 'mDiceI1': tensor(10.5601), 'mAccC': tensor(10.5812), 'mIoUC': tensor(6.1071), 'mDiceC': tensor(11.4990),
                 'mAccCQ': tensor(9.9757), 'mIoUCQ': tensor(5.6032), 'mDiceCQ': tensor(10.6076), 'mAccC1': tensor(9.6963),
                 'mIoUC1': tensor(5.3665), 'mDiceC1': tensor(10.1851)}
        """
        assert len(self.ioufun.tp_list) == len(self.ioufun.tn_list) == len(self.ioufun.fp_list) == len(
            self.ioufun.fn_list) == len(self.ioufun.active_classes_list)
        self.print_log(f"dataset length : {len(self.ioufun.tp_list)}")
        _metrics = self.ioufun.value()
        metrics = {k: round(v.item(), ndigits=4) for k, v in _metrics.items()}

        # 创建一个 PrettyTable 对象
        pt = PrettyTable()
        # 添加表头
        pt.field_names = list(metrics.keys())
        # 添加行数据
        pt.add_row(list(metrics.values()))
        self.print_log('results:')
        self.print_log('\n' + pt.get_string())
        self.ioufun = None
        return metrics

    def evaluate(self, size: int) -> dict:
        """Evaluate the model performance of the whole dataset after processing
        all batches.

        Args:
            size (int): Length of the entire validation dataset. When batch
                size > 1, the dataloader may pad some data samples to make
                sure all ranks have the same length of dataset slice. The
                ``collect_results`` function will drop the padded data based on
                this size.

        Returns:
            dict: Evaluation metrics dict on the val dataset. The keys are the
            names of the metrics, and the values are corresponding results.
        """

        tp_list = collect_results(self.ioufun.tp_list, size, self.collect_device)
        tn_list = collect_results(self.ioufun.tn_list, size, self.collect_device)
        fp_list = collect_results(self.ioufun.fp_list, size, self.collect_device)
        fn_list = collect_results(self.ioufun.fn_list, size, self.collect_device)
        active_classes_list = collect_results(self.ioufun.active_classes_list, size, self.collect_device)
        if is_main_process():
            self.ioufun.tp_list = [i.cpu() for i in tp_list]
            self.ioufun.tn_list = [i.cpu() for i in tn_list]
            self.ioufun.fp_list = [i.cpu() for i in fp_list]
            self.ioufun.fn_list = [i.cpu() for i in fn_list]
            self.ioufun.active_classes_list = [i.cpu() for i in active_classes_list]
            # cast all tensors in results list to cpu

            _metrics = self.compute_metrics(self.results)  # type: ignore
            # Add prefix to metric names
            if self.prefix:
                _metrics = {
                    '/'.join((self.prefix, k)): v
                    for k, v in _metrics.items()
                }
            metrics = [_metrics]
        else:
            metrics = [None]  # type: ignore

        broadcast_object_list(metrics)

        # reset the results list
        self.results.clear()
        return metrics[0]
