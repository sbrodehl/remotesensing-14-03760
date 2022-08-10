# Copyright (C) 2018-2022  Sebastian Brodehl

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


def plot_image(state, tag, img, *args, dataformats="CHW", **kwargs):
    """Add image data to summary.

    Note that this requires the ``pillow`` package.

    Args:
        state (dict): State dict
        tag (string): Data identifier
        img (torch.Tensor, numpy.array, or string/blobname): Image data
        dataformats (string): Image data format specification of the form CHW, HWC, HW, WH, etc.
    Shape:
        img_tensor: Default is `(3, H, W)`. You can use ``torchvision.utils.make_grid()`` to
        convert a batch of tensor into 3xHxW format or call ``add_images`` and let us do the job.
        Tensor with `(1, H, W)`, `(H, W)`, `(H, W, 3)` is also suitable as long as
        corresponding ``dataformats`` argument is passed, e.g. ``CHW``, ``HWC``, ``HW``.

    """
    del args  # unused
    global_step = kwargs.get("global_step", state[state["log.steptype"].value])
    if "global_step" in kwargs:
        del kwargs["global_step"]  # noqa: E701  pylint: disable=C0321
    state["tensorboard"].add_image(
        tag, img, global_step=global_step, dataformats=dataformats, **kwargs
    )


def plot_images(state, tag, imgs, *args, dataformats="NCHW", **kwargs):
    """Add batched image data to summary.

    Note that this requires the ``pillow`` package.

    Args:
        state (dict): State dict
        tag (string): Data identifier
        imgs (torch.Tensor, numpy.array, or string/blobname): Image data
        dataformats (string): Image data format specification of the form
          NCHW, NHWC, CHW, HWC, HW, WH, etc.
    Shape:
        imgs: Default is `(N, 3, H, W)`. If ``dataformats`` is specified,
          other shape will be accepted. e.g. NCHW or NHWC.

    """
    del args  # unused
    global_step = kwargs.get("global_step", state[state["log.steptype"].value])
    if "global_step" in kwargs:
        del kwargs["global_step"]  # noqa: E701  pylint: disable=C0321
    state["tensorboard"].add_images(
        tag, imgs, global_step=global_step, dataformats=dataformats, **kwargs
    )


def plot_image_with_boxes(
    state, tag, img, boxes, *args, labels=None, dataformats="NCHW", **kwargs
):
    """Add image data to summary.

    Note that this requires the ``pillow`` package.

    Args:
        state (dict): State dict
        tag (string): Data identifier
        img (torch.Tensor, numpy.array, or string/blobname): Image data
        boxes (torch.Tensor, numpy.array, or string/blobname): Box data (for detected objects)
          box should be represented as [x1, y1, x2, y2].
        labels (list of string): The label to be shown for each bounding box.
        dataformats (string): Image data format specification of the form
          NCHW, NHWC, CHW, HWC, HW, WH, etc.
    Shape:
        imgs: Default is `(3, H, W)`. It can be specified with ``dataformats`` argument. e.g. CHW or HWC
        boxes: N*4, where N is the number of boxes and each 4 elements in a row represents (xmin, ymin, xmax, ymax).

    """
    del args  # unused
    global_step = kwargs.get("global_step", state[state["log.steptype"].value])
    if "global_step" in kwargs:
        del kwargs["global_step"]  # noqa: E701  pylint: disable=C0321
    state["tensorboard"].add_image_with_boxes(
        tag,
        img,
        boxes,
        global_step=global_step,
        dataformats=dataformats,
        labels=labels,
        **kwargs
    )
