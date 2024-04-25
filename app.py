import io
import base64
from pathlib import Path
import random
import uvicorn
from PIL import Image
from typing import List, Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from anytext_pipeline import AnyTextPipeline
from fastapi.responses import FileResponse


from util import gen_random_seed, process_image_url, save_images, process_base64_image

app = FastAPI()

img_save_folder = "SaveImages"
pipe = AnyTextPipeline(
    ckpt_path="models/anytext_models/anytext_v1.1.ckpt",
    font_path="font/SourceHanSansSC-Medium.otf",
    use_fp16=True,
    device="cuda",
)


class GenerateInput(BaseModel):
    prompt: str = Field('\"没饭\"', title="Prompt", description="提示文本，用于引导图片生成的内容")
    negative_prompt: str = Field("low-res, bad anatomy, extra digit, fewer digits, cropped, worst quality, low quality, watermark, unreadable text, messy words, distorted text, disorganized writing, advertising picture", title="Negative Prompt", description="否定提示，用于避免图片生成中的不期望特征")
    image: str = Field(..., title="Image", description="原始图片，可以是Base64编码的字符串或图片URL")
    masked_image: str = Field(..., title="Masked Image", description="遮罩图片，可以是Base64编码的字符串或图片URL")
    num_inference_steps: int = Field(20, title="Number of Inference Steps", description="推理步数，影响生成图片的细节程度")
    strength: float = Field(1, title="Strength", description="生成强度，用于控制新旧图片内容的结合程度")
    guidance_scale: float = Field(9, title="Guidance Scale", description="引导尺度，用于控制世代中的创造力")
    seed: Optional[int] = Field(None, title="Seed", description="可选的随机种子，用于可复现结果")
    sort_priority: Optional[str] = Field('y', title="Sort Priority", description="排序优先级，用于影响生成结果的排序")
    
class GenerateOutput(BaseModel):
    prompt: str = Field(..., title="Prompt", description="原始提示文本")
    image_base64: List[str] = Field(..., title="Generated Images", description="生成的图片列表，每个图片以Base64编码的字符串形式返回")

@app.post("/generate", response_description="生成的图片以Base64格式返回", summary="生成图片", response_model=GenerateOutput, responses={200: {'description': '图片成功生成并返回Base64编码'}})
async def generate(data: GenerateInput):
    """
    生成图片的接口

    此接口处理随机生成图片的请求。提供提示和否定提示以及相关参数来生成所需的图片。
    
    - **prompt**: 提示文本，用于构成图片的主要内容
    - **negative_prompt**: 否定提示，用于避免生成不想要的图像属性
    - **image**: 可提供原始图片Base64编码字符串或一个URL
    - **masked_image**: 可提供遮罩图片Base64编码字符串或一个URL
    - **num_inference_steps**: 设置推理步骤，较高的值可提供更详细的图片
    - **strength**: 设置图片生成强度
    - **guidance_scale**: 设置引导尺度，控制生成过程中的创造力
    - **seed**: 提供一个种子以获得可复现的结果
    - **sort_priority**: 设置排序优先级，影响结果输出的排序

    如果生成成功，会返回包含图片Base64编码字符串的列表。
    如遇问题，会返回具体的错误信息。
    """
    
    print(data)

    image = data.image
    masked_image = data.masked_image

    if image.startswith('http'):
        image = process_image_url(image)
    else:
        image = process_base64_image(image)
    
    if masked_image.startswith('http'):
        masked_image = process_image_url(masked_image)
    else:
        masked_image = process_base64_image(masked_image)
    
    width = image.shape[1]
    height = image.shape[0]
    
    # results: list of rgb ndarray
    print(f'starting pipeline... {data.prompt}')
    results, rtn_code, rtn_warning = pipe(
        prompt=data.prompt,
        negative_prompt=data.negative_prompt,
        image=image,
        masked_image=masked_image,
        num_inference_steps=data.num_inference_steps,
        strength=data.strength,
        guidance_scale=data.guidance_scale,
        seed=data.seed or gen_random_seed(),
        sort_priority=data.sort_priority,
        width=width,
        height=height
    )
    
    print(f'pipeline done: {rtn_code}, {rtn_warning}')
    
    if rtn_code >= 0:
        save_images(results, img_save_folder)
        print(f"Done, result images are saved in: {img_save_folder}")
    
    else:
        raise HTTPException(status_code=400, detail=rtn_warning)

    # 将生成的图片转换为Base64编码的字符串
    result = []
    for img in results:
        img = Image.fromarray(img)
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        result.append(img_str)
        
    return {"prompt": data.prompt, "image_base64": result}

# 定义根路由
@app.get("/check")
async def hello():
    return {"Hello": "World"}

# 替换basePath为你example_images文件夹的实际路径
basePath = Path(__file__).parent / "example_images"

@app.get("/preview/{file_path:path}")
async def preview(file_path: str):
    # 创建完整的文件路径
    file_location = basePath / file_path

    # 安全地验证路径是否在example_images目录内
    if not str(file_location).startswith(str(basePath)):
        raise HTTPException(status_code=400, detail="Invalid file path")

    # 检查文件是否存在
    if not file_location.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    # 返回文件响应
    return FileResponse(file_location)

def run_uvicorn():
    uvicorn.run(app, host="0.0.0.0", port=8080)
    
if __name__ == "__main__":
    run_uvicorn()
