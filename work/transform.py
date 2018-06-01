import numpy as np
from PIL import Image
import collections
import random
import math
import torchvision.transforms as transforms
import torch
import skimage.transform


class TensorRead:
    def __call__(self, tsr):
        tsr = tsr.permute(2, 0, 1).float().div(255.0)
        return tsr

class RandomVideoCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, vid_tsr):
        b, c, h, w = vid_tsr.size()
        x0 = np.random.randint(0, w-self.size)
        y0 = np.random.randint(0, h-self.size)
        return vid_tsr[:, :, y0:y0+self.size, x0:x0+self.size]

class RandomScaledVideoCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, vid_tsr, min_bound=3/4, max_bound=4/3):
        b, c, h, w = vid_tsr.size()
        r = random.uniform(min_bound, max_bound)
        s = int(self.size*r)
        x0 = np.random.randint(0, w-s)
        y0 = np.random.randint(0, h-s)
        crop_tsr = vid_tsr[:, :, y0:y0+s, x0:x0+s]
        arr_list = [skimage.transform.resize(tsr.numpy(), (c, self.size, self.size), mode='constant', anti_aliasing=True, preserve_range=True) for tsr in crop_tsr]
        tsr_list = [torch.FloatTensor(arr).unsqueeze(0) for arr in arr_list]
        scaled_tsr = torch.cat(tsr_list, dim=0)
        return scaled_tsr

class CenterVideoCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, vid_tsr):
        _, _, h, w = vid_tsr.size()
        x0 = w//2 - self.size//2
        y0 = h//2 - self.size//2
        return vid_tsr[:, :, y0:y0+self.size, x0:x0+self.size].unsqueeze(0)

class CenterVideo5Crop:
    def __init__(self, size):
        self.size = size

    def __call__(self, vid_tsr):
        _, _, h, w = vid_tsr.size()
        hlb = h//2 - self.size//2
        hub = h//2 + self.size//2
        wlb = w//2 - self.size//2
        wub = w//2 + self.size//2
        return torch.cat([
            vid_tsr[:, :, hlb:hub, wlb:wub].unsqueeze(0),
            vid_tsr[:, :, h-self.size:h, w-self.size:w].unsqueeze(0),
            vid_tsr[:, :, 0:self.size, w-self.size:w].unsqueeze(0),
            vid_tsr[:, :, h-self.size:h, 0:self.size].unsqueeze(0),
            vid_tsr[:, :, 0:self.size, 0:self.size].unsqueeze(0),
        ], dim=0)

def _is_pil_image(img):
    return isinstance(img, Image.Image)

def resize(img, size, interpolation=Image.BILINEAR):
    """Resize the input PIL Image to the given size.
    Args:
        img (PIL Image): Image to be resized.
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), the output size will be matched to this. If size is an int,
            the smaller edge of the image will be matched to this number maintaing
            the aspect ratio. i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    Returns:
        PIL Image: Resized image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))
    if not (isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)):
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

def crop(img, i, j, h, w):
    """Crop the given PIL Image.
    Args:
        img (PIL Image): Image to be cropped.
        i: Upper pixel coordinate.
        j: Left pixel coordinate.
        h: Height of the cropped image.
        w: Width of the cropped image.
    Returns:
        PIL Image: Cropped image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    return img.crop((j, i, j + w, i + h))

def resized_crop(img, i, j, h, w, size, interpolation=Image.BILINEAR):
    """Crop the given PIL Image and resize it to desired size.
    Notably used in RandomResizedCrop.
    Args:
        img (PIL Image): Image to be cropped.
        i: Upper pixel coordinate.
        j: Left pixel coordinate.
        h: Height of the cropped image.
        w: Width of the cropped image.
        size (sequence or int): Desired output size. Same semantics as ``scale``.
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``.
    Returns:
        PIL Image: Cropped image.
    """
    assert _is_pil_image(img), 'img should be PIL Image'
    img = crop(img, i, j, h, w)
    img = resize(img, size, interpolation)
    return img

class RandomResizedVideoCrop(object):
    """Crop the given PIL Image to random size and aspect ratio.
    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.
    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR):
        self.size = (size, size)
        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.
        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(*scale) * area
            aspect_ratio = random.uniform(*ratio)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                i = random.randint(0, img.size[1] - h)
                j = random.randint(0, img.size[0] - w)
                return i, j, h, w

        # Fallback
        w = min(img.size[0], img.size[1])
        i = (img.size[1] - w) // 2
        j = (img.size[0] - w) // 2
        return i, j, w, w

    def __call__(self, vid):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.
        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        to_tensor = transforms.ToTensor()
        i, j, h, w = self.get_params(vid[0], self.scale, self.ratio)
        img_list = [resized_crop(img, i, j, h, w, self.size, self.interpolation) for img in vid]
        vid_tsr = torch.cat([to_tensor(img).unsqueeze(0) for img in img_list])
        return vid_tsr

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string


class CenterVideoCropImage:
    def __init__(self, size):
        self.size = size

    def __call__(self, img_list):
        w, h = img_list[0].size
        x0 = w//2
        y0 = h//2
        to_tensor = transforms.ToTensor()
        img_list = [img.crop((x0, y0, w, h)) for img in img_list]
        vid_tsr = torch.cat([to_tensor(img).unsqueeze(0) for img in img_list])
        return vid_tsr
