import gradio as gr
import torch
import traceback
import torch
from PIL import Image
import sys
import os
import time
from typing import List, Dict, Optional, Tuple
import json
from datetime import datetime

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion
import cv2

# 添加系统路径
sys.path.append(".")
sys.path.append("..")

# 导入必要的模块
from clip_pipeline_attend_and_excite import RelationalAttendAndExcitePipeline
from config import RunConfig
from run import run_on_prompt_and_masked_image, get_indices_to_alter_new
from utils.ptp_utils import AttentionStore
from utils import vis_utils

# English Translations
localization_texts = {
    "en": {
        "title": "Anomaly Any",
        "description": "This is a demo for generating anomaly images. You can specify the object type and anomaly description, and the model will generate an image with the specified anomaly.",
        "step1_title": "Step 1: Define Prompts via Templates",
        "type_label": "Type",
        "type_placeholder": "e.g., hazelnut",
        "normal_prompt_template_label": "Normal Prompt Template",
        "normal_prompt_template_placeholder": "e.g., a photo of a {type}",
        "prompt_template_label": "Anomaly Prompt Template",
        "prompt_template_placeholder": "e.g., a photo of a {type} with a crack on it",
        "detailed_prompt_template_label": "Detailed Prompt Template",
        "detailed_prompt_template_placeholder": "e.g., a photo of a {type} with a large, jagged crack exposing the interior.",
        "prompt_preview_header": "✨ Prompt Previews",
        "normal_prompt_preview_label": "Normal Prompt Preview",
        "anomaly_prompt_preview_label": "Anomaly Prompt Preview",
        "detailed_prompt_preview_label": "Detailed Prompt Preview",
        "step2_title": "Step 2: Upload Reference Image (Normal)",
        "masking_options": "Optional: Masking",
        "mask_image": "Mask Image (Optional)",
        "mask_ratio": "Mask Size (%)",
        "generate_mask": "Generate Random Mask",
        "advanced_settings": "Optional: Advanced Settings",
        "token_indices": "Token Indices",
        "token_indices_placeholder": "e.g., 8 or 5,6,7",
        "random_seed": "Random Seed",
        "image_guidance_strength": "Image Guidance Strength",
        "scale_factor": "Scale Factor for Attention",
        "threshold_step": "Threshold at Step {}",
        "max_iter": "Max Iterations to Alter",
        "use_standard_sd": "Use Standard SD (for comparison)",
        "generate_btn": "Generate Anomaly Image",
        "output_results": "Output",
        "generated_image": "Generated Image",
        "attention_map": "Attention Map",
        "generation_info": "Generation Info",
        "quick_examples": "Quick Examples",
        "generation_time": "Generation Time: {:.2f} seconds",
        "used_prompt": "Used Prompt: {}",
        "token_indices_info": "Token Indices: {}",
        "error_prefix": "Error: {}",
        "processing": "Processing...",
        "ready": "Ready",
        "model_loading": "Loading model...",
        "model_loaded": "Model loaded successfully!",
        "random_mask": "Random Mask Generation",
        "mask_ratio": "Mask Size Ratio",
        "generate_mask": "Generate Random Mask",
        "mask_generated": "Random mask generated successfully!",
        "token_attention_maps": "Token Attention Maps",
    }
}

# Global Variables
NUM_DIFFUSION_STEPS = 200
GUIDANCE_SCALE = 12.5
MAX_NUM_WORDS = 77
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

# Custom CSS Styles
custom_css = """
/* Overall style optimization */
.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    max-width: 1400px !important;
    margin: auto !important;
}

/* Title style */
h1 {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
    margin-bottom: 0.5em;
}

/* Card style */
.gr-group {
    border: 1px solid rgba(0, 0, 0, 0.1);
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 16px;
    background: rgba(255, 255, 255, 0.6);
    backdrop-filter: blur(10px);
    transition: all 0.3s ease;
}

.gr-group:hover {
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08);
    transform: translateY(-2px);
}

/* Button style */
.gr-button-primary {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 12px 24px !important;
    border-radius: 8px !important;
    transition: all 0.3s ease !important;
}

.gr-button-primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(102, 126, 234, 0.4) !important;
}

/* Input box style */
.gr-textbox, .gr-number, .gr-slider {
    border-radius: 8px !important;
    border: 1px solid rgba(0, 0, 0, 0.1) !important;
    transition: all 0.3s ease !important;
}

.gr-textbox:focus, .gr-number:focus {
    border-color: #667eea !important;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
}

/* Image upload area */
.gr-image {
    border-radius: 12px !important;
    overflow: hidden !important;
    border: 2px dashed rgba(0, 0, 0, 0.2) !important;
    transition: all 0.3s ease !important;
}

.gr-image:hover {
    border-color: #667eea !important;
}

/* Accordion style */
.gr-accordion {
    border-radius: 12px !important;
    border: 1px solid rgba(0, 0, 0, 0.1) !important;
    overflow: hidden !important;
    margin-top: 16px !important;
}

.gr-accordion-header {
    background: rgba(255, 255, 255, 0.8) !important;
    padding: 12px 16px !important;
}

/* Progress bar style */
.progress-bar {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
}

/* Label style */
label {
    font-weight: 600 !important;
    color: #334155 !important;
    margin-bottom: 8px !important;
}

/* Example style */
.gr-examples {
    border-radius: 12px !important;
    background: rgba(255, 255, 255, 0.6) !important;
    padding: 16px !important;
}

/* Dark mode support */
@media (prefers-color-scheme: dark) {
    .gr-group {
        background: rgba(30, 30, 30, 0.6);
        border-color: rgba(255, 255, 255, 0.1);
    }
    
    label {
        color: #e2e8f0 !important;
    }
}

/* Responsive design */
@media (max-width: 768px) {
    .gradio-container {
        padding: 16px !important;
    }
    
    .gr-group {
        padding: 12px;
    }
}

/* Animation effect */
@keyframes fadeIn {
    from {
        opacity: 0;
        transform: translateY(10px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.gr-group, .gr-image, .gr-textbox {
    animation: fadeIn 0.5s ease-out;
}

#attention_gallery_centered {
    margin-left: auto;
    margin-right: auto;
    max-width: 90%; /* Ensure it doesn't overflow small screens, adjust as needed */
}
/* Center images within each gallery item */
#attention_gallery_centered .thumbnail-item {
    display: flex;
    justify-content: center; /* Horizontally center the image within the item */
    align-items: center;    /* Vertically center the image (optional, but good for consistency) */
}
#attention_gallery_centered .thumbnail-item > img {
    max-width: 100%; /* Ensure image does not overflow its container */
    max-height: 100%;
    object-fit: contain; /* Already set in component, but good to be explicit */
}

/* Make seed input and button look more integrated */
#seed_row .gr-number {
    border-top-right-radius: 0px !important;
    border-bottom-right-radius: 0px !important;
    border-right-width: 0px !important; /* Remove right border */
}
#seed_row button {
    border-top-left-radius: 0px !important;
    border-bottom-left-radius: 0px !important;
    margin-left: -1px; /* Overlap slightly to merge borders */
}
#seed_row .gr-number input {
    text-align: left !important; /* Ensure number input text is left-aligned */
}
"""

def _generate_shape(width, height):
    """Helper function to generate a random base shape."""
    mask = np.zeros((height, width), dtype=np.uint8)
    shape_type = np.random.choice(['ellipse', 'polygon', 'line'])

    if shape_type == 'ellipse':
        num_ellipses = np.random.randint(2, 5)
        for _ in range(num_ellipses):
            center_x, center_y = np.random.randint(0, width), np.random.randint(0, height)
            is_elongated = np.random.rand() > 0.5
            if is_elongated:
                axis_1, axis_2 = np.random.randint(width // 10, width // 2), np.random.randint(5, 20)
                if np.random.rand() > 0.5: # Randomly swap axes
                    axis_1, axis_2 = axis_2, axis_1
            else:
                axis_1, axis_2 = np.random.randint(width // 8, width // 4), np.random.randint(height // 8, height // 4)
            angle = np.random.randint(0, 360)
            cv2.ellipse(mask, (center_x, center_y), (axis_1, axis_2), angle, 0, 360, 255, -1)

    elif shape_type == 'polygon':
        num_vertices = np.random.randint(4, 10)
        vertices = np.random.randint(0, width, (num_vertices, 2))
        # Ensure vertices are ordered to form a non-self-intersecting polygon
        center = np.mean(vertices, axis=0)
        angles = np.arctan2(vertices[:, 1] - center[1], vertices[:, 0] - center[0])
        vertices = vertices[np.argsort(angles)]
        cv2.fillPoly(mask, [vertices], 255)

    elif shape_type == 'line':
        num_lines = np.random.randint(3, 8)
        for _ in range(num_lines):
            x1, y1 = np.random.randint(0, width), np.random.randint(0, height)
            x2, y2 = np.random.randint(0, width), np.random.randint(0, height)
            thickness = np.random.randint(5, 25)
            cv2.line(mask, (x1, y1), (x2, y2), 255, thickness)

    return mask

def generate_random_mask(mask_ratio_percent: float) -> Tuple[Image.Image, str]:
    """
    Generates a random, complex mask with a precise area ratio.
    """
    try:
        mask_ratio = mask_ratio_percent / 100.0
        width, height = 512, 512

        # 1. Generate a base shape using one of the random strategies
        base_mask = _generate_shape(width, height)

        # 2. Ensure single connected component
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(base_mask, 4, cv2.CV_32S)
        if num_labels <= 1: return generate_random_mask(mask_ratio_percent) # Retry if empty
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = np.where(labels == largest_label, 255, 0).astype(np.uint8)

        # 3. Precisely scale the mask to the target ratio
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return generate_random_mask(mask_ratio_percent) # Retry

        cnt = max(contours, key=cv2.contourArea)
        current_area = cv2.contourArea(cnt)
        target_area = width * height * mask_ratio

        if current_area > 10: # Ensure base shape is not too small
            scale_factor = np.sqrt(target_area / current_area)
            M = cv2.moments(cnt)
            cx = int(M['m10'] / M['m00']) if M['m00'] != 0 else width // 2
            cy = int(M['m01'] / M['m00']) if M['m00'] != 0 else height // 2

            cnt_final = ((cnt - [cx, cy]) * scale_factor + [cx, cy]).astype(np.int32)

            # 4. Draw final mask and apply post-processing
            final_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.drawContours(final_mask, [cnt_final], -1, 255, -1)

            # Add random noise/blur for more natural edges
            kernel_size = np.random.randint(7, 21) // 2 * 2 + 1
            final_mask = cv2.GaussianBlur(final_mask, (kernel_size, kernel_size), 0)
            _, final_mask = cv2.threshold(final_mask, np.random.randint(100, 150), 255, cv2.THRESH_BINARY)
            
            final_area = np.sum(final_mask > 0)
            info_text = f"Mask generated successfully! Target: {mask_ratio_percent:.2f}%, Actual: {100 * final_area / (width * height):.2f}%"
            
            return Image.fromarray(final_mask, mode='L'), info_text
        else:
            return generate_random_mask(mask_ratio_percent) # Retry for better base shape

    except Exception as e:
        return None, f"Error generating mask: {str(e)}"

def load_examples():
    """Loads a hardcoded list of examples."""
    
    base_dir = os.path.abspath(os.path.dirname(__file__))
    example_image_dir = os.path.join(base_dir, "example_image")
    example_web_dir = os.path.join(base_dir, "example_web")
    mask_dir = os.path.join(example_image_dir, "fg_mask")
    default_normal_template = "a photo of a {type}"

    examples = [
        [
            "hazelnut",
            default_normal_template,
            "a photo of a {type} with a crack on it",
            "a photo of a {type} with a large, jagged crack exposing the interior.",
            8,
            os.path.join(example_image_dir, "000.png"),
            os.path.join(mask_dir, "000.png"),
            14291
        ],
        [
            "table",
            default_normal_template,
            "a photo of a {type} that is faded",
            "a photo of a {type} with areas of discoloration or lightening due to prolonged sun exposure",
            8,
            os.path.join(example_web_dir, "table.jpg"),
            None,
            45678
        ],
        [
            "road",
            default_normal_template,
            "a photo of a {type} with cracks",
            "a photo of a {type} with visible fractures and deteriorating surface",
            7,
            os.path.join(example_image_dir, "road.png"),
            None,
            56789
        ]
    ]
    
    # Verify that files exist to prevent Gradio from erroring on missing files
    verified_examples = []
    for ex in examples:
        image_path = ex[5] # Index adjusted for new normal_prompt_template
        mask_path = ex[6]  # Index adjusted
        if image_path and os.path.exists(image_path):
            if mask_path and not os.path.exists(mask_path):
                print(f"Warning: Mask file not found for example {ex[0]}: {mask_path}. Setting mask to None.")
                ex[6] = None # Index adjusted
            verified_examples.append(ex)
        else:
            print(f"Warning: Image file not found for example {ex[0]}: {image_path}. Skipping example.")

    return verified_examples

def get_text(key: str) -> str:
    """Get text from translation dictionary"""
    return localization_texts["en"].get(key, key)

# Initialize model
print("Loading model...")
stable = RelationalAttendAndExcitePipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", 
    safety_checker=None
).to(device)
tokenizer = stable.tokenizer
print("Model loaded successfully!")

def run_and_display(prompts: List[str],
                    controller: AttentionStore,
                    indices_to_alter: List[int],
                    init_image: Optional[Image.Image],
                    init_image_guidance_scale: float,
                    mask_image: Optional[str],
                    generator: torch.Generator,
                    run_standard_sd: bool = False,
                    scale_factor: int = 20,
                    thresholds: Dict[int, float] = {0: 0.05, 10: 0.5, 20: 0.8},
                    max_iter_to_alter: int = 25,
                    normal_prompt: str = "",
                    detailed_prompt: str = "",
                    progress: gr.Progress = None) -> Tuple[Image.Image, torch.Tensor]:
    """Run anomaly generation pipeline"""
    
    # Progress callback
    def progress_callback(step, timestep, latents):
        if progress is not None:
            progress(step / NUM_DIFFUSION_STEPS, desc=f"Diffusion Step {step}/{NUM_DIFFUSION_STEPS}")

    config = RunConfig(
        prompt=prompts[0],
        run_standard_sd=run_standard_sd,
        scale_factor=scale_factor,
        thresholds=thresholds,
        max_iter_to_alter=max_iter_to_alter,
        n_inference_steps=NUM_DIFFUSION_STEPS,
        guidance_scale=GUIDANCE_SCALE
    )
    
    image, image_latent = run_on_prompt_and_masked_image(
        model=stable,
        prompt=prompts,
        controller=controller,
        token_indices=indices_to_alter,
        init_image=init_image,
        init_image_guidance_scale=init_image_guidance_scale,
        mask_image=mask_image,
        seed=generator,
        config=config,
        normal_prompt=normal_prompt,
        detailed_prompt=detailed_prompt,
        callback=progress_callback,
        callback_steps=1
    )
    
    return image, image_latent

def generate_anomaly_image(
    type_str: str,
    normal_prompt_template: str,
    prompt_template: str,
    detailed_prompt_template: str,
    token_indices_str: str,
    normal_image: Image.Image,
    mask_image: Optional[Image.Image],
    seed: int,
    init_image_guidance_scale: float,
    scale_factor: int,
    threshold_0: float,
    threshold_10: float,
    threshold_20: float,
    max_iter_to_alter: int,
    use_standard_sd: bool,
    progress=gr.Progress()
) -> Tuple[Image.Image, Image.Image, str]:
    """Generate anomaly image main function"""
    
    try:
        progress(0, desc="Initializing...")
        
        # Build prompt
        prompt = prompt_template.format(type=type_str)
        normal_prompt = normal_prompt_template.format(type=type_str)
        detailed_prompt = detailed_prompt_template.format(type=type_str)
        
        # Handle guidance image
        if normal_image is None:
            return None, None, "Error: Please upload a normal guidance image"
        
        # Resize image
        normal_image = normal_image.convert("RGB")
        normal_image.thumbnail((512, 512))
        
        # Handle mask image
        mask_path = None
        if mask_image is not None:
            mask_path = f"temp_mask_{datetime.now().timestamp()}.png"
            mask_image = mask_image.convert("L")
            mask_image.thumbnail((512, 512))
            mask_image.save(mask_path)
        
        progress(0.05, desc="Parsing Tokens...")
        
        # Get token indices
        indices_to_alter, token_map = get_indices_to_alter_new(stable, prompt, str(token_indices_str)) # Ensure it's a string
        
        # Set random seed
        generator = torch.Generator(device).manual_seed(seed)
        
        # Create controller
        controller = AttentionStore()
        
        # Set thresholds
        thresholds = {
            0: threshold_0,
            10: threshold_10,
            20: threshold_20
        }
        
        # Record start time
        start_time = time.time()
        
        # Generate image
        image, image_latent = run_and_display(
            prompts=[prompt],
            controller=controller,
            indices_to_alter=indices_to_alter,
            init_image=normal_image,
            init_image_guidance_scale=init_image_guidance_scale,
            mask_image=mask_path,
            generator=generator,
            run_standard_sd=use_standard_sd,
            scale_factor=scale_factor,
            thresholds=thresholds,
            max_iter_to_alter=max_iter_to_alter,
            normal_prompt=normal_prompt,
            detailed_prompt=detailed_prompt,
            progress=progress
        )
        
        progress(0.9, desc="Generating Attention Maps...")
        
        # Generate attention map
        attention_images = vis_utils.show_cross_attention(
            attention_store=controller,
            prompt=prompt,
            tokenizer=tokenizer,
            res=16,
            from_where=("up", "down", "mid"),
            indices_to_alter=indices_to_alter,
            orig_image=image
        )
        
        # Clean up temporary files
        if mask_path and os.path.exists(mask_path):
            os.remove(mask_path)
        
        # Calculate generation time
        generation_time = time.time() - start_time
        
        progress(1.0, desc="Ready")
        
        # Format token map for display
        token_map_str = "\n".join([f"{idx}: {word}" for idx, word in token_map.items()])
        
        info_text = f"""
Generation Successful!
Generation Time: {generation_time:.2f} seconds
Used Prompt: {prompt}
Altered Token Indices: {token_indices_str}

---
**Token Map:**
{token_map_str}
"""
        
        return image, attention_images, info_text
        
    except Exception as e:
        traceback.print_exc()
        return None, None, f"Error: {str(e)}"

def update_prompt_previews(type_str, normal_prompt_template, prompt_template, detailed_template):
    """Dynamically update the prompt preview textboxes."""
    try:
        if not type_str or not type_str.strip():
             raise ValueError("Type is empty")
        normal_prompt = normal_prompt_template.format(type=type_str)
        anomaly_prompt = prompt_template.format(type=type_str)
        detailed_prompt = detailed_template.format(type=type_str)
    except (KeyError, IndexError, ValueError) as e:
        normal_prompt = "Waiting for valid type..."
        anomaly_prompt = f"Error: {e}"
        detailed_prompt = f"Error: {e}"
    return normal_prompt, anomaly_prompt, detailed_prompt

def build_ui():
    # --- Custom CSS ---
    custom_css = """
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap');
    .gradio-container {
        font-family: 'Inter', sans-serif;
        max-width: 1280px !important; 
        margin: auto !important;
    }
    .gr-button {white-space: nowrap;}
    #refresh-seed-btn {max-width: 2.5em; min-width: 2.5em !important;}
    .gr-group {border-radius: 10px !important; box-shadow: 0 2px 5px rgba(0,0,0,0.05) !important;}
    h1, h3 {letter-spacing: -0.5px;}
    """

    default_type = "hazelnut"
    default_normal_prompt_template = "a photo of a {type}"
    default_prompt_template = "a photo of a {type} with a crack on it"
    default_detailed_template = "a photo of a {type} with a large, jagged crack exposing the interior."
    default_token_indices = "8"
    default_mask_ratio = 5.0 # Corresponds to slider's initial value if it were 0-100, but it's 0.1-25

    with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
        gr.Markdown(f"# {get_text('title')}\n\n{get_text('description')}")

        with gr.Row(equal_height=False):
            # --- Input Column ---
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown(f"### {get_text('step1_title')}")
                    type_textbox = gr.Textbox(
                        label=get_text("type_label"),
                        placeholder=get_text("type_placeholder"),
                        value=default_type
                    )
                    normal_prompt_template_textbox = gr.Textbox(
                        label=get_text("normal_prompt_template_label"),
                        placeholder=get_text("normal_prompt_template_placeholder"),
                        value=default_normal_prompt_template
                    )
                    prompt_template_textbox = gr.Textbox(
                        label=get_text("prompt_template_label"),
                        placeholder=get_text("prompt_template_placeholder"),
                        value=default_prompt_template
                    )
                    detailed_prompt_template_textbox = gr.Textbox(
                        label=get_text("detailed_prompt_template_label"),
                        placeholder=get_text("detailed_prompt_template_placeholder"),
                        value=default_detailed_template
                    )

                with gr.Group():
                    gr.Markdown(f"### {get_text('prompt_preview_header')}")
                    normal_prompt_preview_display = gr.Textbox(label=get_text('normal_prompt_preview_label'), interactive=False)
                    anomaly_prompt_preview_display = gr.Textbox(label=get_text('anomaly_prompt_preview_label'), interactive=False)
                    detailed_prompt_preview_display = gr.Textbox(label=get_text('detailed_prompt_preview_label'), interactive=False)

                with gr.Group():
                    gr.Markdown(f"### {get_text('step2_title')}")
                    normal_image_input = gr.Image(label=get_text("normal_image"), type="pil") # Changed from normal_image to step2_title for label consistency

                with gr.Accordion(get_text("masking_options"), open=False):
                    mask_image_input = gr.Image(label=get_text("mask_image"), type="pil")
                    with gr.Row():
                        mask_ratio_slider = gr.Slider(label=get_text("mask_ratio"), minimum=0.1, maximum=25.0, value=default_mask_ratio, step=0.1)
                        generate_mask_button = gr.Button(get_text("generate_mask"), variant="secondary", size="sm")
                    mask_info_display = gr.Textbox(label="Mask Info", visible=False, interactive=False) # Renamed from mask_info for clarity

                with gr.Accordion(get_text("advanced_settings"), open=False):
                    token_indices_textbox = gr.Number(
                        label=get_text("token_indices"),
                        value=int(default_token_indices) if default_token_indices else 0, # Convert default string to int
                        precision=0, # Integer input
                        step=1, # Increment/decrement by 1
                        minimum=0 # Assuming token indices are non-negative
                    )
                    with gr.Row(variant='compact'):
                        with gr.Row(elem_id="seed_row", variant='compact'): # Use compact variant
                            seed_number = gr.Number(
                                label=get_text("random_seed"),
                                value=np.random.randint(0, 2**32 - 1),
                                precision=0,
                                elem_id="seed_input",
                                scale=4 # Adjust scale
                            )
                            random_seed_button = gr.Button("🎲", elem_id="random_seed_button_inline", scale=1, min_width=40) # Adjust scale and min_width
                    guidance_slider = gr.Slider(label=get_text("image_guidance_strength"), minimum=0.1, maximum=1.0, value=0.3, step=0.05)
                    scale_slider = gr.Slider(label=get_text("scale_factor"), minimum=10, maximum=100, value=20, step=5)
                    threshold_0_slider = gr.Slider(label=get_text("threshold_step").format(0), minimum=0.01, maximum=0.5, value=0.05, step=0.01)
                    threshold_10_slider = gr.Slider(label=get_text("threshold_step").format(10), minimum=0.1, maximum=0.9, value=0.5, step=0.05)
                    threshold_20_slider = gr.Slider(label=get_text("threshold_step").format(20), minimum=0.1, maximum=0.9, value=0.8, step=0.05)
                    max_iter_slider = gr.Slider(label=get_text("max_iter"), minimum=5, maximum=50, value=25, step=5)
                    standard_sd_checkbox = gr.Checkbox(label=get_text("use_standard_sd"), value=False)

                with gr.Row():
                    clear_button = gr.Button("Clear")
                    generate_button = gr.Button(get_text("generate_btn"), variant="primary", scale=2)

            # --- Output Column ---
            with gr.Column(scale=1):
                gr.Markdown(f"### {get_text('output_results')}")
                main_anomaly_image_display = gr.Image(label=get_text("generated_image"), type="pil", interactive=False)
                attention_map_gallery = gr.Gallery(
                    label=get_text("token_attention_maps"), 
                    columns=2, 
                    object_fit="contain", 
                    height=450, 
                    elem_id="attention_gallery_centered" # For CSS centering
                )
                info_textbox_display = gr.Textbox(label=get_text("generation_info"), lines=5, interactive=False) # Renamed

        gr.Markdown(f"### {get_text('quick_examples')}")
        
        # Define component lists for clarity and reusability
        all_template_inputs = [type_textbox, normal_prompt_template_textbox, prompt_template_textbox, detailed_prompt_template_textbox]
        preview_outputs = [normal_prompt_preview_display, anomaly_prompt_preview_display, detailed_prompt_preview_display]
        
        main_inputs = all_template_inputs + [
            token_indices_textbox, normal_image_input, mask_image_input, seed_number
        ]
        adv_inputs = [
            guidance_slider, scale_slider, threshold_0_slider, threshold_10_slider, 
            threshold_20_slider, max_iter_slider, standard_sd_checkbox
        ]
        generation_outputs = [main_anomaly_image_display, attention_map_gallery, info_textbox_display]

        example_loader_inputs = all_template_inputs + [token_indices_textbox, normal_image_input, mask_image_input, seed_number]

        gr.Examples(
            examples=load_examples(),
            inputs=example_loader_inputs,
            label=None,
            examples_per_page=5
        )

        # --- Event Handlers ---
        for t_input in all_template_inputs:
            t_input.change(update_prompt_previews, inputs=all_template_inputs, outputs=preview_outputs, queue=False)
        
        demo.load(update_prompt_previews, inputs=all_template_inputs, outputs=preview_outputs, queue=False)

        generate_button.click(
            fn=generate_anomaly_image,
            inputs=main_inputs + adv_inputs,
            outputs=generation_outputs,
            api_name="generate"
        )

        def clear_all_fn():
            new_seed = np.random.randint(0, 2**32 - 1)
            initial_normal_preview, initial_anomaly_preview, initial_detailed_preview = update_prompt_previews(
                default_type, default_normal_prompt_template, default_prompt_template, default_detailed_template
            )
            return (
                default_type, default_normal_prompt_template, default_prompt_template, default_detailed_template, 
                default_token_indices, None, None, default_mask_ratio, new_seed, # Main inputs (type, templates, token, images, mask_ratio, seed)
                None, None, "", # Output fields (output_image, attention_map, info_textbox)
                "", # mask_info_display
                initial_normal_preview, initial_anomaly_preview, initial_detailed_preview # Preview fields
            )
        
        components_to_clear = main_inputs[:-1] + [mask_ratio_slider] + [main_inputs[-1]] + generation_outputs + [mask_info_display] + preview_outputs
        # main_inputs includes seed_number at the end. mask_ratio_slider is separate.
        # Order for clear_all_fn outputs:
        # type_textbox, normal_prompt_template_textbox, prompt_template_textbox, detailed_prompt_template_textbox,
        # token_indices_textbox, normal_image_input, mask_image_input, mask_ratio_slider, seed_number,
        # output_image_display, attention_map_gallery, info_textbox_display,
        # mask_info_display,
        # normal_prompt_preview_display, anomaly_prompt_preview_display, detailed_prompt_preview_display

        clear_button.click(
            fn=clear_all_fn,
            inputs=None, 
            outputs=components_to_clear
        )

        random_seed_button.click(lambda: gr.update(value=np.random.randint(0, 2**32 - 1)), outputs=seed_number)
        generate_mask_button.click(generate_random_mask, inputs=[mask_ratio_slider], outputs=[mask_image_input, mask_info_display])

    return demo

if __name__ == "__main__":
    torch.cuda.empty_cache()
    
    app_demo = build_ui()
    app_demo.queue(max_size=3)
    app_demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        debug=False,
        show_error=True,
        favicon_path=None
    )