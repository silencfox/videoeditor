from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

def gen_simple_frame(text, out_path: Path, size=(512, 512)):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, color=(30, 30, 40))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 20)
    except Exception:
        font = ImageFont.load_default()
    draw.text((10, 10), text, font=font, fill=(230, 230, 230))
    img.save(out_path)
    return out_path

def _generate_sd_frame(prompt: str, out_path: Path, steps=15, guidance=6.5, width=448, height=448, model_id="runwayml/stable-diffusion-v1-5"):
    try:
        from diffusers import StableDiffusionPipeline
        import torch
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
        pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
        image = pipe(prompt, num_inference_steps=steps, guidance_scale=guidance, height=height, width=width).images[0]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(out_path)
        return out_path
    except Exception as e:
        print("Stable Diffusion failed or not available:", e)
        return gen_simple_frame(prompt[:200], out_path, size=(width, height))

def generate_images_from_script(script: str, out_dir: Path, frames_per_scene=8, use_sd=False, internet_ok=True, sd_steps=15, sd_guidance=6.5, sd_width=448, sd_height=448, sd_model_id="runwayml/stable-diffusion-v1-5"):
    scenes = [s.strip() for s in script.split("\n\n") if s.strip()]
    frame_paths = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for si, scene in enumerate(scenes):
        for f in range(frames_per_scene):
            path = out_dir / f"scene{si:03d}_frame{f:04d}.png"
            if use_sd:
                _generate_sd_frame(f"{scene} cinematic, high quality", path, steps=sd_steps, guidance=sd_guidance, width=sd_width, height=sd_height, model_id=sd_model_id)
            else:
                gen_simple_frame(scene[:200], path, size=(sd_width, sd_height))
            frame_paths.append(path)
    return frame_paths
