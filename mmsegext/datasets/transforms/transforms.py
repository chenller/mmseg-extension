from mmcv.transforms.base import BaseTransform
from mmseg.registry import TRANSFORMS

from mmcv.transforms.processing import Pad as mmcv_Pad
@TRANSFORMS.register_module('ext-PadOnlyImg')
class PadOnlyImg(mmcv_Pad):
    """Pad the image & segmentation map.

        There are three padding modes: (1) pad to a fixed size and (2) pad to the
        minimum size that is divisible by some number. and (3)pad to square. Also,
        pad to square and pad to the minimum size can be used as the same time.

        Required Keys:

        - img

        Modified Keys:

        - img

        Added Keys:

        - pad_shape
        - pad_fixed_size
        - pad_size_divisor

        Args:
            size (tuple, optional): Fixed padding size.
                Expected padding shape (w, h). Defaults to None.
            size_divisor (int, optional): The divisor of padded size. Defaults to
                None.
            pad_to_square (bool): Whether to pad the image into a square.
                Currently only used for YOLOX. Defaults to False.
            pad_val (Number | dict[str, Number], optional): Padding value for if
                the pad_mode is "constant". If it is a single number, the value
                to pad the image is the number and to pad the semantic
                segmentation map is 255. If it is a dict, it should have the
                following keys:

                - img: The value to pad the image.
                - seg: The value to pad the semantic segmentation map.

                Defaults to dict(img=0, seg=255).
            padding_mode (str): Type of padding. Should be: constant, edge,
                reflect or symmetric. Defaults to 'constant'.

                - constant: pads with a constant value, this value is specified
                  with pad_val.
                - edge: pads with the last value at the edge of the image.
                - reflect: pads with reflection of image without repeating the last
                  value on the edge. For example, padding [1, 2, 3, 4] with 2
                  elements on both sides in reflect mode will result in
                  [3, 2, 1, 2, 3, 4, 3, 2].
                - symmetric: pads with reflection of image repeating the last value
                  on the edge. For example, padding [1, 2, 3, 4] with 2 elements on
                  both sides in symmetric mode will result in
                  [2, 1, 1, 2, 3, 4, 4, 3]
        """
    def transform(self, results: dict) -> dict:
        """Call function to pad images, masks, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Updated result dict.
        """
        self._pad_img(results)
        return results