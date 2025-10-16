import torch, json
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel

@torch.no_grad()
def run_inference_single(image_path: str, model_dir: str, device="cuda"):
    """
    Run inference using your fine-tuned Donut model.
    Returns: (fields_dict, raw_ocr_text)
    """
    print(f"🔍 Running Donut inference for {image_path}")
    processor = DonutProcessor.from_pretrained(model_dir)
    model = VisionEncoderDecoderModel.from_pretrained(model_dir).to(device)
    model.eval()

    image = Image.open(image_path).convert("RGB")
    task_prompt = "<s_receipt>"
    decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

    outputs = model.generate(
        pixel_values,
        decoder_input_ids=decoder_input_ids,
        max_length=256,
        return_dict_in_generate=True
    )

    seq = processor.batch_decode(outputs.sequences)[0]
    seq = seq.replace("<s_receipt>", "").replace("</s>", "").strip()

    try:
        fields = json.loads(seq)
    except Exception:
        print("⚠️ Could not parse output JSON — returning raw text.")
        fields = {"raw_output": seq}

    return fields, seq
