import math
import numbers
import warnings
from collections.abc import Iterable

import numpy as np
import torch
from PIL import Image, __version__ as PILLOW_VERSION
from numpy import sin, cos, tan
from torch import Tensor
from torch.jit.annotations import List

try:
    import accimage
except ImportError:
    accimage = None

import custom_functional_pil as F_pil
import custom_functional_tensor as F_t


def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)


def _is_numpy(img):
    return isinstance(img, np.ndarray)


def _is_numpy_image(img):
    return img.ndim in {2, 3}


def to_tensor(pic):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    See ``ToTensor`` for more details.

    Args:
        pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

    Returns:
        Tensor: Converted image.
    """
    if not (_is_pil_image(pic) or _is_numpy(pic)):
        raise TypeError('pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

    if _is_numpy(pic) and not _is_numpy_image(pic):
        raise ValueError('pic should be 2/3 dimensional. Got {} dimensions.'.format(pic.ndim))

    if isinstance(pic, np.ndarray):
        # handle numpy array
        if pic.ndim == 2:
            pic = pic[:, :, None]

        img = torch.from_numpy(pic.transpose((2, 0, 1)))
        # backward compatibility
        if isinstance(img, torch.ByteTensor):
            return img.float().div(255)
        else:
            return img

    if accimage is not None and isinstance(pic, accimage.Image):
        nppic = np.zeros([pic.channels, pic.height, pic.width], dtype=np.float32)
        pic.copyto(nppic)
        return torch.from_numpy(nppic)

    # handle PIL Image
    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    elif pic.mode == 'F':
        img = torch.from_numpy(np.array(pic, np.float32, copy=False))
    elif pic.mode == '1':
        img = 255 * torch.from_numpy(np.array(pic, np.uint8, copy=False))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))

    img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
    # put it from HWC to CHW format
    img = img.permute((2, 0, 1)).contiguous()
    if isinstance(img, torch.ByteTensor):
        return img.float().div(255)
    else:
        return img


def pil_to_tensor(pic):
    """Convert a ``PIL Image`` to a tensor of the same type.

    See ``AsTensor`` for more details.

    Args:
        pic (PIL Image): Image to be converted to tensor.

    Returns:
        Tensor: Converted image.
    """
    if not (_is_pil_image(pic)):
        raise TypeError('pic should be PIL Image. Got {}'.format(type(pic)))

    if accimage is not None and isinstance(pic, accimage.Image):
        nppic = np.zeros([pic.channels, pic.height, pic.width], dtype=np.float32)
        pic.copyto(nppic)
        return torch.as_tensor(nppic)

    # handle PIL Image
    img = torch.as_tensor(np.asarray(pic))
    img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
    # put it from HWC to CHW format
    img = img.permute((2, 0, 1))
    return img


def convert_image_dtype(image: torch.Tensor, dtype: torch.dtype = torch.float) -> torch.Tensor:
    """Convert a tensor image to the given ``dtype`` and scale the values accordingly

    Args:
        image (torch.Tensor): Image to be converted
        dtype (torch.dtype): Desired datasets type of the output

    Returns:
        (torch.Tensor): Converted image

    .. note::

        When converting from a smaller to a larger integer ``dtype`` the maximum values are **not** mapped exactly.
        If converted back and forth, this mismatch has no effect.

    Raises:
        RuntimeError: When trying to cast :class:`torch.float32` to :class:`torch.int32` or :class:`torch.int64` as
            well as for trying to cast :class:`torch.float64` to :class:`torch.int64`. These conversions might lead to
            overflow errors since the floating point ``dtype`` cannot store consecutive integers over the whole range
            of the integer ``dtype``.
    """
    if image.dtype == dtype:
        return image

    if image.dtype.is_floating_point:
        # float to float
        if dtype.is_floating_point:
            return image.to(dtype)

        # float to int
        if (image.dtype == torch.float32 and dtype in (torch.int32, torch.int64)) or (
                image.dtype == torch.float64 and dtype == torch.int64
        ):
            msg = f"The cast from {image.dtype} to {dtype} cannot be performed safely."
            raise RuntimeError(msg)

        eps = 1e-3
        return image.mul(torch.iinfo(dtype).max + 1 - eps).to(dtype)
    else:
        # int to float
        if dtype.is_floating_point:
            max = torch.iinfo(image.dtype).max
            image = image.to(dtype)
            return image / max

        # int to int
        input_max = torch.iinfo(image.dtype).max
        output_max = torch.iinfo(dtype).max

        if input_max > output_max:
            factor = (input_max + 1) // (output_max + 1)
            image = image // factor
            return image.to(dtype)
        else:
            factor = (output_max + 1) // (input_max + 1)
            image = image.to(dtype)
            return image * factor


def to_pil_image(pic, mode=None):
    """Convert a tensor or an ndarray to PIL Image.

    See :class:`~torchvision.transforms.ToPILImage` for more details.

    Args:
        pic (Tensor or numpy.ndarray): Image to be converted to PIL Image.
        mode (`PIL.Image mode`_): color space and pixel depth of input datasets (optional).

    .. _PIL.Image mode: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#concept-modes

    Returns:
        PIL Image: Image converted to PIL Image.
    """
    if not (isinstance(pic, torch.Tensor) or isinstance(pic, np.ndarray)):
        raise TypeError('pic should be Tensor or ndarray. Got {}.'.format(type(pic)))

    elif isinstance(pic, torch.Tensor):
        if pic.ndimension() not in {2, 3}:
            raise ValueError('pic should be 2/3 dimensional. Got {} dimensions.'.format(pic.ndimension()))

        elif pic.ndimension() == 2:
            # if 2D image, add channel dimension (CHW)
            pic = pic.unsqueeze(0)

    elif isinstance(pic, np.ndarray):
        if pic.ndim not in {2, 3}:
            raise ValueError('pic should be 2/3 dimensional. Got {} dimensions.'.format(pic.ndim))

        elif pic.ndim == 2:
            # if 2D image, add channel dimension (HWC)
            pic = np.expand_dims(pic, 2)

    npimg = pic
    if isinstance(pic, torch.FloatTensor) and mode != 'F':
        pic = pic.mul(255).byte()
    if isinstance(pic, torch.Tensor):
        npimg = np.transpose(pic.numpy(), (1, 2, 0))

    if not isinstance(npimg, np.ndarray):
        raise TypeError('Input pic must be a torch.Tensor or NumPy ndarray, ' +
                        'not {}'.format(type(npimg)))

    if npimg.shape[2] == 1:
        expected_mode = None
        npimg = npimg[:, :, 0]
        if npimg.dtype == np.uint8:
            expected_mode = 'L'
        elif npimg.dtype == np.int16:
            expected_mode = 'I;16'
        elif npimg.dtype == np.int32:
            expected_mode = 'I'
        elif npimg.dtype == np.float32:
            expected_mode = 'F'
        if mode is not None and mode != expected_mode:
            raise ValueError("Incorrect mode ({}) supplied for input type {}. Should be {}"
                             .format(mode, np.dtype, expected_mode))
        mode = expected_mode

    elif npimg.shape[2] == 2:
        permitted_2_channel_modes = ['LA']
        if mode is not None and mode not in permitted_2_channel_modes:
            raise ValueError("Only modes {} are supported for 2D inputs".format(permitted_2_channel_modes))

        if mode is None and npimg.dtype == np.uint8:
            mode = 'LA'

    elif npimg.shape[2] == 4:
        permitted_4_channel_modes = ['RGBA', 'CMYK', 'RGBX']
        if mode is not None and mode not in permitted_4_channel_modes:
            raise ValueError("Only modes {} are supported for 4D inputs".format(permitted_4_channel_modes))

        if mode is None and npimg.dtype == np.uint8:
            mode = 'RGBA'
    else:
        permitted_3_channel_modes = ['RGB', 'YCbCr', 'HSV']
        if mode is not None and mode not in permitted_3_channel_modes:
            raise ValueError("Only modes {} are supported for 3D inputs".format(permitted_3_channel_modes))
        if mode is None and npimg.dtype == np.uint8:
            mode = 'RGB'

    if mode is None:
        raise TypeError('Input type {} is not supported'.format(npimg.dtype))

    return Image.fromarray(npimg, mode=mode)


def normalize(tensor, mean, std, inplace=False):
    """Normalize a tensor image with mean and standard deviation.

    .. note::
        This transform acts out of place by default, i.e., it does not mutates the input tensor.

    See :class:`~torchvision.transforms.Normalize` for more details.

    Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation inplace.

    Returns:
        Tensor: Normalized Tensor image.
    """
    if not torch.is_tensor(tensor):
        raise TypeError('tensor should be a torch tensor. Got {}.'.format(type(tensor)))

    if tensor.ndimension() != 3:
        raise ValueError('Expected tensor to be a tensor image of size (C, H, W). Got tensor.size() = '
                         '{}.'.format(tensor.size()))

    if not inplace:
        tensor = tensor.clone()

    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    if (std == 0).any():
        raise ValueError('std evaluated to zero after conversion to {}, leading to division by zero.'.format(dtype))
    if mean.ndim == 1:
        mean = mean[:, None, None]
    if std.ndim == 1:
        std = std[:, None, None]
    tensor.sub_(mean).div_(std)
    return tensor


def resize(img, size, interpolation=Image.BILINEAR):
    r"""Resize the input PIL Image to the given size.

    Args:
        img (PIL Image): Image to be resized.
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), the output size will be matched to this. If size is an int,
            the smaller edge of the image will be matched to this number maintaining
            the aspect ratio. i.e, if height > width, then image will be rescaled to
            :math:`\left(\text{size} \times \frac{\text{height}}{\text{width}}, \text{size}\right)`
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``

    Returns:
        PIL Image: Resized image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))
    if not (isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)):
        raise TypeError('Got inappropriate size arg: {}'.format(size))

    if isinstance(size, int):
        w, h = img.size
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            return img.resize((ow, oh), interpolation)
        else:
            oh = size
            ow = int(size * w / h)
            return img.resize((ow, oh), interpolation)
    else:
        return img.resize(size[::-1], interpolation)


def scale(*args, **kwargs):
    warnings.warn("The use of the transforms.Scale transform is deprecated, " +
                  "please use transforms.Resize instead.")
    return resize(*args, **kwargs)


def pad(img: Tensor, padding: List[int], fill: int = 0, padding_mode: str = "constant") -> Tensor:
    r"""Pad the given image on all sides with the given "pad" value.
    The image can be a PIL Image or a torch Tensor, in which case it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions

    Args:
        img (PIL Image or Tensor): Image to be padded.
        padding (int or tuple or list): Padding on each border. If a single int is provided this
            is used to pad all borders. If tuple of length 2 is provided this is the padding
            on left/right and top/bottom respectively. If a tuple of length 4 is provided
            this is the padding for the left, top, right and bottom borders respectively.
            In torchscript mode padding as single int is not supported, use a tuple or
            list of length 1: ``[padding, ]``.
        fill (int or str or tuple): Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant. Only int value is supported for Tensors.
        padding_mode: Type of padding. Should be: constant, edge, reflect or symmetric. Default is constant.
            Only "constant" is supported for Tensors as of now.

            - constant: pads with a constant value, this value is specified with fill

            - edge: pads with the last value on the edge of the image

            - reflect: pads with reflection of image (without repeating the last value on the edge)

                       padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                       will result in [3, 2, 1, 2, 3, 4, 3, 2]

            - symmetric: pads with reflection of image (repeating the last value on the edge)

                         padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                         will result in [2, 1, 1, 2, 3, 4, 4, 3]

    Returns:
        PIL Image or Tensor: Padded image.
    """
    if not isinstance(img, torch.Tensor):
        return F_pil.pad(img, padding=padding, fill=fill, padding_mode=padding_mode)

    return F_t.pad(img, padding=padding, fill=fill, padding_mode=padding_mode)


def crop(img, top, left, height, width):
    """Crop the given PIL Image.

    Args:
        img (PIL Image): Image to be cropped. (0,0) denotes the top left corner of the image.
        top (int): Vertical component of the top left corner of the crop box.
        left (int): Horizontal component of the top left corner of the crop box.
        height (int): Height of the crop box.
        width (int): Width of the crop box.

    Returns:
        PIL Image: Cropped image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    return img.crop((left, top, left + width, top + height))


def center_crop(img, output_size):
    """Crop the given PIL Image and resize it to desired size.

    Args:
        img (PIL Image): Image to be cropped. (0,0) denotes the top left corner of the image.
        output_size (sequence or int): (height, width) of the crop box. If int,
            it is used for both directions
    Returns:
        PIL Image: Cropped image.
    """
    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))
    image_width, image_height = img.size
    crop_height, crop_width = output_size
    crop_top = int(round((image_height - crop_height) / 2.))
    crop_left = int(round((image_width - crop_width) / 2.))
    return crop(img, crop_top, crop_left, crop_height, crop_width)


def resized_crop(img, top, left, height, width, size, interpolation=Image.BILINEAR):
    """Crop the given PIL Image and resize it to desired size.

    Notably used in :class:`~torchvision.transforms.RandomResizedCrop`.

    Args:
        img (PIL Image): Image to be cropped. (0,0) denotes the top left corner of the image.
        top (int): Vertical component of the top left corner of the crop box.
        left (int): Horizontal component of the top left corner of the crop box.
        height (int): Height of the crop box.
        width (int): Width of the crop box.
        size (sequence or int): Desired output size. Same semantics as ``resize``.
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``.
    Returns:
        PIL Image: Cropped image.
    """
    assert _is_pil_image(img), 'img should be PIL Image'
    img = crop(img, top, left, height, width)
    img = resize(img, size, interpolation)
    return img


def hflip(img: Tensor) -> Tensor:
    """Horizontally flip the given PIL Image or torch Tensor.

    Args:
        img (PIL Image or Torch Tensor): Image to be flipped. If img
            is a Tensor, it is expected to be in [..., H, W] format,
            where ... means it can have an arbitrary number of trailing
            dimensions.

    Returns:
        PIL Image:  Horizontally flipped image.
    """
    if not isinstance(img, torch.Tensor):
        return F_pil.hflip(img)

    return F_t.hflip(img)


def _parse_fill(fill, img, min_pil_version):
    """Helper function to get the fill color for rotate and perspective transforms.

    Args:
        fill (n-tuple or int or float): Pixel fill value for area outside the transformed
            image. If int or float, the value is used for all bands respectively.
            Defaults to 0 for all bands.
        img (PIL Image): Image to be filled.
        min_pil_version (str): The minimum PILLOW version for when the ``fillcolor`` option
            was first introduced in the calling function. (e.g. rotate->5.2.0, perspective->5.0.0)

    Returns:
        dict: kwarg for ``fillcolor``
    """
    major_found, minor_found = (int(v) for v in PILLOW_VERSION.split('.')[:2])
    major_required, minor_required = (int(v) for v in min_pil_version.split('.')[:2])
    if major_found < major_required or (major_found == major_required and minor_found < minor_required):
        if fill is None:
            return {}
        else:
            msg = ("The option to fill background area of the transformed image, "
                   "requires pillow>={}")
            raise RuntimeError(msg.format(min_pil_version))

    num_bands = len(img.getbands())
    if fill is None:
        fill = 0
    if isinstance(fill, (int, float)) and num_bands > 1:
        fill = tuple([fill] * num_bands)
    if not isinstance(fill, (int, float)) and len(fill) != num_bands:
        msg = ("The number of elements in 'fill' does not match the number of "
               "bands of the image ({} != {})")
        raise ValueError(msg.format(len(fill), num_bands))

    return {"fillcolor": fill}


def _get_perspective_coeffs(startpoints, endpoints):
    """Helper function to get the coefficients (a, b, c, d, e, f, g, h) for the perspective transforms.

    In Perspective Transform each pixel (x, y) in the original image gets transformed as,
     (x, y) -> ( (ax + by + c) / (gx + hy + 1), (dx + ey + f) / (gx + hy + 1) )

    Args:
        List containing [top-left, top-right, bottom-right, bottom-left] of the original image,
        List containing [top-left, top-right, bottom-right, bottom-left] of the transformed
                   image
    Returns:
        octuple (a, b, c, d, e, f, g, h) for transforming each pixel.
    """
    matrix = []

    for p1, p2 in zip(endpoints, startpoints):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])

    A = torch.tensor(matrix, dtype=torch.float)
    B = torch.tensor(startpoints, dtype=torch.float).view(8)
    res = torch.lstsq(B, A)[0]
    return res.squeeze_(1).tolist()


def perspective(img, startpoints, endpoints, interpolation=Image.BICUBIC, fill=None):
    """Perform perspective transform of the given PIL Image.

    Args:
        img (PIL Image): Image to be transformed.
        startpoints: List containing [top-left, top-right, bottom-right, bottom-left] of the original image
        endpoints: List containing [top-left, top-right, bottom-right, bottom-left] of the transformed image
        interpolation: Default- Image.BICUBIC
        fill (n-tuple or int or float): Pixel fill value for area outside the rotated
            image. If int or float, the value is used for all bands respectively.
            This option is only available for ``pillow>=5.0.0``.

    Returns:
        PIL Image:  Perspectively transformed Image.
    """

    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    opts = _parse_fill(fill, img, '5.0.0')

    coeffs = _get_perspective_coeffs(startpoints, endpoints)
    return img.transform(img.size, Image.PERSPECTIVE, coeffs, interpolation, **opts)


def vflip(img: Tensor) -> Tensor:
    """Vertically flip the given PIL Image or torch Tensor.

    Args:
        img (PIL Image or Torch Tensor): Image to be flipped. If img
            is a Tensor, it is expected to be in [..., H, W] format,
            where ... means it can have an arbitrary number of trailing
            dimensions.

    Returns:
        PIL Image:  Vertically flipped image.
    """
    if not isinstance(img, torch.Tensor):
        return F_pil.vflip(img)

    return F_t.vflip(img)


def five_crop(img, size):
    """Crop the given PIL Image into four corners and the central crop.

    .. Note::
        This transform returns a tuple of images and there may be a
        mismatch in the number of inputs and targets your ``Dataset`` returns.

    Args:
       size (sequence or int): Desired output size of the crop. If size is an
           int instead of sequence like (h, w), a square crop (size, size) is
           made.

    Returns:
       tuple: tuple (tl, tr, bl, br, center)
                Corresponding top left, top right, bottom left, bottom right and center crop.
    """
    if isinstance(size, numbers.Number):
        size = (int(size), int(size))
    else:
        assert len(size) == 2, "Please provide only two dimensions (h, w) for size."

    image_width, image_height = img.size
    crop_height, crop_width = size
    if crop_width > image_width or crop_height > image_height:
        msg = "Requested crop size {} is bigger than input size {}"
        raise ValueError(msg.format(size, (image_height, image_width)))

    tl = img.crop((0, 0, crop_width, crop_height))
    tr = img.crop((image_width - crop_width, 0, image_width, crop_height))
    bl = img.crop((0, image_height - crop_height, crop_width, image_height))
    br = img.crop((image_width - crop_width, image_height - crop_height,
                   image_width, image_height))
    center = center_crop(img, (crop_height, crop_width))
    return (tl, tr, bl, br, center)


def ten_crop(img, size, vertical_flip=False):
    """Generate ten cropped images from the given PIL Image.
    Crop the given PIL Image into four corners and the central crop plus the
    flipped version of these (horizontal flipping is used by default).

    .. Note::
        This transform returns a tuple of images and there may be a
        mismatch in the number of inputs and targets your ``Dataset`` returns.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        vertical_flip (bool): Use vertical flipping instead of horizontal

    Returns:
        tuple: tuple (tl, tr, bl, br, center, tl_flip, tr_flip, bl_flip, br_flip, center_flip)
            Corresponding top left, top right, bottom left, bottom right and
            center crop and same for the flipped image.
    """
    if isinstance(size, numbers.Number):
        size = (int(size), int(size))
    else:
        assert len(size) == 2, "Please provide only two dimensions (h, w) for size."

    first_five = five_crop(img, size)

    if vertical_flip:
        img = vflip(img)
    else:
        img = hflip(img)

    second_five = five_crop(img, size)
    return first_five + second_five


def adjust_brightness(img: Tensor, brightness_factor: float) -> Tensor:
    """Adjust brightness of an Image.

    Args:
        img (PIL Image or Torch Tensor): Image to be adjusted.
        brightness_factor (float):  How much to adjust the brightness. Can be
            any non negative number. 0 gives a black image, 1 gives the
            original image while 2 increases the brightness by a factor of 2.

    Returns:
        PIL Image or Torch Tensor: Brightness adjusted image.
    """
    if not isinstance(img, torch.Tensor):
        return F_pil.adjust_brightness(img, brightness_factor)

    return F_t.adjust_brightness(img, brightness_factor)


def adjust_contrast(img: Tensor, contrast_factor: float) -> Tensor:
    """Adjust contrast of an Image.

    Args:
        img (PIL Image or Torch Tensor): Image to be adjusted.
        contrast_factor (float): How much to adjust the contrast. Can be any
            non negative number. 0 gives a solid gray image, 1 gives the
            original image while 2 increases the contrast by a factor of 2.

    Returns:
        PIL Image or Torch Tensor: Contrast adjusted image.
    """
    if not isinstance(img, torch.Tensor):
        return F_pil.adjust_contrast(img, contrast_factor)

    return F_t.adjust_contrast(img, contrast_factor)


def adjust_saturation(img: Tensor, saturation_factor: float) -> Tensor:
    """Adjust color saturation of an image.

    Args:
        img (PIL Image or Torch Tensor): Image to be adjusted.
        saturation_factor (float):  How much to adjust the saturation. 0 will
            give a black and white image, 1 will give the original image while
            2 will enhance the saturation by a factor of 2.

    Returns:
        PIL Image or Torch Tensor: Saturation adjusted image.
    """
    if not isinstance(img, torch.Tensor):
        return F_pil.adjust_saturation(img, saturation_factor)

    return F_t.adjust_saturation(img, saturation_factor)


def adjust_hue(img: Tensor, hue_factor: float) -> Tensor:
    """Adjust hue of an image.

    The image hue is adjusted by converting the image to HSV and
    cyclically shifting the intensities in the hue channel (H).
    The image is then converted back to original image mode.

    `hue_factor` is the amount of shift in H channel and must be in the
    interval `[-0.5, 0.5]`.

    See `Hue`_ for more details.

    .. _Hue: https://en.wikipedia.org/wiki/Hue

    Args:
        img (PIL Image): PIL Image to be adjusted.
        hue_factor (float):  How much to shift the hue channel. Should be in
            [-0.5, 0.5]. 0.5 and -0.5 give complete reversal of hue channel in
            HSV space in positive and negative direction respectively.
            0 means no shift. Therefore, both -0.5 and 0.5 will give an image
            with complementary colors while 0 gives the original image.

    Returns:
        PIL Image: Hue adjusted image.
    """
    if not isinstance(img, torch.Tensor):
        return F_pil.adjust_hue(img, hue_factor)

    raise TypeError('img should be PIL Image. Got {}'.format(type(img)))


def adjust_gamma(img, gamma, gain=1):
    r"""Perform gamma correction on an image.

    Also known as Power Law Transform. Intensities in RGB mode are adjusted
    based on the following equation:

    .. math::
        I_{\text{out}} = 255 \times \text{gain} \times \left(\frac{I_{\text{in}}}{255}\right)^{\gamma}

    See `Gamma Correction`_ for more details.

    .. _Gamma Correction: https://en.wikipedia.org/wiki/Gamma_correction

    Args:
        img (PIL Image): PIL Image to be adjusted.
        gamma (float): Non negative real number, same as :math:`\gamma` in the equation.
            gamma larger than 1 make the shadows darker,
            while gamma smaller than 1 make dark regions lighter.
        gain (float): The constant multiplier.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    if gamma < 0:
        raise ValueError('Gamma should be a non-negative real number')

    input_mode = img.mode
    img = img.convert('RGB')

    gamma_map = [255 * gain * pow(ele / 255., gamma) for ele in range(256)] * 3
    img = img.point(gamma_map)  # use PIL's point-function to accelerate this part

    img = img.convert(input_mode)
    return img


def rotate(img, angle, resample=False, expand=False, center=None, fill=None):
    """Rotate the image by angle.


    Args:
        img (PIL Image): PIL Image to be rotated.
        angle (float or int): In degrees degrees counter clockwise order.
        resample (``PIL.Image.NEAREST`` or ``PIL.Image.BILINEAR`` or ``PIL.Image.BICUBIC``, optional):
            An optional resampling filter. See `filters`_ for more information.
            If omitted, or if the image has mode "1" or "P", it is set to ``PIL.Image.NEAREST``.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output image to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
        fill (n-tuple or int or float): Pixel fill value for area outside the rotated
            image. If int or float, the value is used for all bands respectively.
            Defaults to 0 for all bands. This option is only available for ``pillow>=5.2.0``.

    .. _filters: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters

    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    opts = _parse_fill(fill, img, '5.2.0')

    return img.rotate(angle, resample, expand, center, **opts)


def _get_inverse_affine_matrix(center, angle, translate, scale, shear):
    # Helper method to compute inverse matrix for affine transformation

    # As it is explained in PIL.Image.rotate
    # We need compute INVERSE of affine transformation matrix: M = T * C * RSS * C^-1
    # where T is translation matrix: [1, 0, tx | 0, 1, ty | 0, 0, 1]
    #       C is translation matrix to keep center: [1, 0, cx | 0, 1, cy | 0, 0, 1]
    #       RSS is rotation with scale and shear matrix
    #       RSS(a, s, (sx, sy)) =
    #       = R(a) * S(s) * SHy(sy) * SHx(sx)
    #       = [ s*cos(a - sy)/cos(sy), s*(-cos(a - sy)*tan(x)/cos(y) - sin(a)), 0 ]
    #         [ s*sin(a + sy)/cos(sy), s*(-sin(a - sy)*tan(x)/cos(y) + cos(a)), 0 ]
    #         [ 0                    , 0                                      , 1 ]
    #
    # where R is a rotation matrix, S is a scaling matrix, and SHx and SHy are the shears:
    # SHx(s) = [1, -tan(s)] and SHy(s) = [1      , 0]
    #          [0, 1      ]              [-tan(s), 1]
    #
    # Thus, the inverse is M^-1 = C * RSS^-1 * C^-1 * T^-1

    if isinstance(shear, numbers.Number):
        shear = [shear, 0]

    if not isinstance(shear, (tuple, list)) and len(shear) == 2:
        raise ValueError(
            "Shear should be a single value or a tuple/list containing " +
            "two values. Got {}".format(shear))

    rot = math.radians(angle)
    sx, sy = [math.radians(s) for s in shear]

    cx, cy = center
    tx, ty = translate

    # RSS without scaling
    a = cos(rot - sy) / cos(sy)
    b = -cos(rot - sy) * tan(sx) / cos(sy) - sin(rot)
    c = sin(rot - sy) / cos(sy)
    d = -sin(rot - sy) * tan(sx) / cos(sy) + cos(rot)

    # Inverted rotation matrix with scale and shear
    # det([[a, b], [c, d]]) == 1, since det(rotation) = 1 and det(shear) = 1
    M = [d, -b, 0,
         -c, a, 0]
    M = [x / scale for x in M]

    # Apply inverse of translation and of center translation: RSS^-1 * C^-1 * T^-1
    M[2] += M[0] * (-cx - tx) + M[1] * (-cy - ty)
    M[5] += M[3] * (-cx - tx) + M[4] * (-cy - ty)

    # Apply center translation: C * RSS^-1 * C^-1 * T^-1
    M[2] += cx
    M[5] += cy
    return M


def affine(img, angle, translate, scale, shear, resample=0, fillcolor=None):
    """Apply affine transformation on the image keeping image center invariant

    Args:
        img (PIL Image): PIL Image to be rotated.
        angle (float or int): rotation angle in degrees between -180 and 180, clockwise direction.
        translate (list or tuple of integers): horizontal and vertical translations (post-rotation translation)
        scale (float): overall scale
        shear (float or tuple or list): shear angle value in degrees between -180 to 180, clockwise direction.
        If a tuple of list is specified, the first value corresponds to a shear parallel to the x axis, while
        the second value corresponds to a shear parallel to the y axis.
        resample (``PIL.Image.NEAREST`` or ``PIL.Image.BILINEAR`` or ``PIL.Image.BICUBIC``, optional):
            An optional resampling filter.
            See `filters`_ for more information.
            If omitted, or if the image has mode "1" or "P", it is set to ``PIL.Image.NEAREST``.
        fillcolor (int): Optional fill color for the area outside the transform in the output image. (Pillow>=5.0.0)
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
        "Argument translate should be a list or tuple of length 2"

    assert scale > 0.0, "Argument scale should be positive"

    output_size = img.size
    center = (img.size[0] * 0.5 + 0.5, img.size[1] * 0.5 + 0.5)
    matrix = _get_inverse_affine_matrix(center, angle, translate, scale, shear)
    kwargs = {"fillcolor": fillcolor} if int(PILLOW_VERSION.split('.')[0]) >= 5 else {}
    return img.transform(output_size, Image.AFFINE, matrix, resample, **kwargs)


def to_grayscale(img, num_output_channels=1):
    """Convert image to grayscale version of image.

    Args:
        img (PIL Image): Image to be converted to grayscale.

    Returns:
        PIL Image: Grayscale version of the image.
            if num_output_channels = 1 : returned image is single channel

            if num_output_channels = 3 : returned image is 3 channel with r = g = b
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    if num_output_channels == 1:
        img = img.convert('L')
    elif num_output_channels == 3:
        img = img.convert('L')
        np_img = np.array(img, dtype=np.uint8)
        np_img = np.dstack([np_img, np_img, np_img])
        img = Image.fromarray(np_img, 'RGB')
    else:
        raise ValueError('num_output_channels should be either 1 or 3')

    return img


def erase(img, i, j, h, w, v, inplace=False):
    """ Erase the input Tensor Image with given value.

    Args:
        img (Tensor Image): Tensor image of size (C, H, W) to be erased
        i (int): i in (i,j) i.e coordinates of the upper left corner.
        j (int): j in (i,j) i.e coordinates of the upper left corner.
        h (int): Height of the erased region.
        w (int): Width of the erased region.
        v: Erasing value.
        inplace(bool, optional): For in-place operations. By default is set False.

    Returns:
        Tensor Image: Erased image.
    """
    if not isinstance(img, torch.Tensor):
        raise TypeError('img should be Tensor Image. Got {}'.format(type(img)))

    if not inplace:
        img = img.clone()

    img[:, i:i + h, j:j + w] = v
    return img
