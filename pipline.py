import torch
import torchvision.transforms as transforms
import os
import PIL.Image as Image
from model import createDeepLabv3


class GrayToRGBAcolor(object):
    """ 四通道灰度图片tensor转RGBA
        Args:
            snr （list[int,int,int,float]）: [r,g,b,a] <[0:255,0:255,0:255,0.0:1.0]>
        """

    def __init__(self, rgba):
        self.rgb = (torch.Tensor(rgba) / 255.0).unsqueeze(dim=-1).unsqueeze(dim=-1)
        self.a = rgba[-1]

    def __call__(self, img):
        new_img = img * self.rgb
        new_img[-1, :, :] = self.a
        return new_img


class blackTransparent(object):
    """ 令黑色（0，0，0，x）的像素透明，变成（0，0，0，0）
    """

    def __call__(self, img):
        index = (img[0] == 0) * (img[0] == 0) * (img[0] == 0)
        new_image = torch.zeros_like(img, dtype=torch.float)
        new_image.copy_(img)
        new_image[-1, index] = 0
        return new_image


class pipline:
    def __init__(self):
        self.model = None
        self.threshold = 0.5
        self.device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")

    def prepare(self, th=0.5):
        self.threshold = th
        self.model = createDeepLabv3(exp_directory="CFExp", inherit=True)
        self.model.to(self.device)
        self.model.eval()

    def __call__(self, images, binary='both'):
        if self.model is None:
            raise ValueError("Model not initialized. Please call 'prepare' first.")

        if isinstance(images, list) and all(isinstance(img, Image.Image) for img in images):
            tenser_images = [transforms.ToTensor()(img).unsqueeze(0).to(self.device) for img in images]
        elif isinstance(images, torch.Tensor):
            tenser_images = images.to(self.device)
        else:
            raise TypeError("Images must be a list of PIL.Image or a torch.Tensor.")

        predictions = []
        with torch.no_grad():
            if isinstance(tenser_images, list):
                for img in tenser_images:
                    output = self.model(img)
                    pre = output['out']
                    if binary == 'both':
                        pre = (pre > self.threshold).float()
                    elif binary == 'up':
                        pre = torch.where(condition=(pre > self.threshold), input=pre, other=torch.zeros_like(pre))
                    else:
                        pre = (pre - pre.min()) / (pre.max() - pre.min()) * 255
                        pre = pre.clamp(0, 255).type(torch.uint8)
                        pre = pre.squeeze(dim=0)
                    predictions.append(pre)
            elif isinstance(tenser_images, torch.Tensor):
                output = self.model(tenser_images)
                pre = output['out']
                if binary == 'both':
                    pre = (pre > self.threshold).float()
                elif binary == 'up':
                    pre = torch.where(condition=(pre > self.threshold), input=pre, other=torch.zeros_like(pre))
                return pre
        pil_out = []
        rgba_out = []
        to_pil = transforms.ToPILImage()
        to_mask = transforms.Compose([transforms.Lambda(lambda x: x.repeat(4, 1, 1) if x.size(0) == 1 else x),
                                      GrayToRGBAcolor([128, 0, 128, 0.4]),
                                      blackTransparent(),
                                      transforms.ToPILImage()
                                      ])
        for pre, origin in zip(predictions, images):
            pre = pre.squeeze().cpu()
            pil_image = to_pil(pre)
            mask = to_mask(pre)
            rgba_image = Image.alpha_composite(origin.convert('RGBA'), mask)

            pil_out.append(pil_image)
            rgba_out.append(rgba_image)

        return pil_out, rgba_out


if __name__ == "__main__":
    pipline_instance = pipline()
    th_ = 0.2
    pipline_instance.prepare(th_)

    example_image_folder_path = r"C:\Users\QianQichen\Desktop\大创\专用数据集"

#  r"C:\Users\QianQichen\Desktop\大创\DeepLabv3\CrackForest\Images"
#  r"C:\Users\QianQichen\Desktop\大创\Crack_Segmentation_Dataset\Images"
#  r"C:\Users\QianQichen\Desktop\大创\专用数据集"
    images_to_process = []
    count = 0
    names = os.listdir(example_image_folder_path)
    for pic_name in names:
        if pic_name.endswith(".jpg") or pic_name.endswith(".png"):

            img_path = os.path.join(example_image_folder_path, pic_name)

            img_ = Image.open(img_path)
            images_to_process.append(img_)
            count += 1
            if count > 100:
                break
    # Process the collected images
    segmented = pipline_instance(images_to_process, binary='up')

    # Save segmented images

    for name, seg in zip(names, segmented[0]):
        seg.save(
            fr"C:\Users\QianQichen\Desktop\大创\专用数据集推理结果\二值化\_{th_}_segmented_{name}.png"
        )
    for name, seg_mark in zip(names,segmented[1]):
        seg_mark.save(
            fr"C:\Users\QianQichen\Desktop\大创\专用数据集推理结果\叠加\_{th_}_segmented_marked_{name}.png"
        )
# fr"C:\Users\QianQichen\Desktop\大创\DeepLabv3\segmented\_{th_}_segmented_{idx}.png"
# fr"C:\Users\QianQichen\Desktop\大创\DeepLabv3\segmented\_{th_}_segmented_marked_{idx}.png"

# fr"C:\Users\QianQichen\Desktop\大创\new\segmented\_{th_}_segmented_{idx}.png"
# fr"C:\Users\QianQichen\Desktop\大创\new\segmented\_{th_}_segmented_marked_{idx}.png"


# fr"C:\Users\QianQichen\Desktop\大创\专用数据集推理结果\二值化\_{th_}_segmented_{idx}.png"
# fr"C:\Users\QianQichen\Desktop\大创\专用数据集推理结果\叠加\_{th_}_segmented_marked_{idx}.png"
