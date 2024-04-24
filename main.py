import cv2
import os

from anytext_pipeline import AnyTextPipeline
from utils import save_images

seed = 66273235

pipe = AnyTextPipeline(
    ckpt_path="model/anytext/anytext_v1.1_fp16.ckpt",
    font_path="font/SourceHanSansSC-Medium.otf",
    use_fp16=False,
    device="mps",
)

img_save_folder = "SaveImages"
rgb_image = cv2.imread(
    "example_images/ref2.jpg"
)[..., ::-1]

masked_image = cv2.imread(
    "example_images/edit2.png"
)[..., ::-1]

rgb_image = cv2.resize(rgb_image, (512, 512))
masked_image = cv2.resize(masked_image, (512, 512))

results, rtn_code, rtn_warning = pipe(
    prompt='"爆炸"',
    negative_prompt="low-res, bad anatomy, extra digit, fewer digits, cropped, worst quality, low quality, watermark, unreadable text, messy words, distorted text, disorganized writing, advertising picture",
    image=rgb_image,
    masked_image=masked_image,
    num_inference_steps=20,
    strength=1.0,
    guidance_scale=9.0,
    height=rgb_image.shape[0],
    width=rgb_image.shape[1],
    seed=seed,
    sort_priority="y",
)
if rtn_code >= 0:
    save_images(results, img_save_folder)
    print(f"Done, result images are saved in: {img_save_folder}")
