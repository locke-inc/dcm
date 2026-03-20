"""
test_generate.py — Test DCM generation from a checkpoint
=========================================================
Usage (Kaggle):
    !python /kaggle/working/dcm/test_generate.py
"""

from __future__ import annotations

import torch
from dcm_model import DCMConfig, DiffusionContextModel


def main():
    # Must match the config used during training
    cfg = DCMConfig(
        latent_dim=512,
        num_latent_vectors=16,
        denoiser_hidden_dim=256,
    )

    print("Loading model...")
    model = DiffusionContextModel(cfg, device_map="auto")

    # Load trained weights
    ckpt_path = "/kaggle/working/dcm_checkpoints/checkpoint-final/dcm_training_state.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.encoder.load_state_dict(ckpt["encoder"])
    model.diffuser.load_state_dict(ckpt["diffuser"])
    model.decoder.memory_proj.load_state_dict(ckpt["memory_proj"])

    # Load LoRA adapter
    from peft import PeftModel
    lora_path = "/kaggle/working/dcm_checkpoints/checkpoint-final/lora_adapter"
    model.decoder.model = PeftModel.from_pretrained(
        model.decoder.model, lora_path
    )

    model.eval()
    tokenizer = model.decoder.tokenizer
    device = model._device

    # --- Test cases ---
    test_contexts = [
        "It was a dark and stormy night. The wind howled through the ancient trees surrounding the old manor house. Inside, the fire crackled softly as the old man sat in his chair, staring into the flames. He had lived in this house for sixty years, and every corner held a memory.",
        "The experiment had yielded unexpected results. Dr. Chen stared at the data on her screen, her coffee growing cold beside her keyboard. The protein folding patterns were unlike anything previously documented in the literature. She reached for her phone to call her colleague.",
        "In the beginning, the universe was a singularity of infinite density. Then came the expansion, rapid and violent, flinging matter and energy across the void. Billions of years passed. Stars formed, burned, and died. From their ashes, new stars were born, and around them, planets coalesced from dust and gas.",
    ]

    prompts = [
        "He slowly rose from his chair and",
        "The implications of this discovery",
        "On one such planet, life emerged",
    ]

    print("\n" + "=" * 60)
    print("  DCM Generation Test")
    print("=" * 60)

    for i, (context, prompt) in enumerate(zip(test_contexts, prompts)):
        print(f"\n--- Test {i+1} ---")
        print(f"Context: {context[:100]}...")
        print(f"Prompt:  {prompt}")

        context_ids = tokenizer.encode(context, return_tensors="pt").to(device)
        prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            output_ids = model.generate_with_memory(
                context_ids=context_ids,
                prompt_ids=prompt_ids,
                max_new_tokens=64,
                temperature=0.8,
                diffusion_steps=50,
            )

        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"Output:  {generated_text}")

    # --- Comparison: same prompt WITHOUT memory (random noise as memory) ---
    print("\n" + "=" * 60)
    print("  Control: Random memory (no real context encoding)")
    print("=" * 60)

    prompt = "He slowly rose from his chair and"
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Feed garbage context to see if real context actually matters
    garbage_context = "asdf qwerty zxcv 1234 bnm poiu lkjh gfds"
    garbage_ids = tokenizer.encode(garbage_context, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate_with_memory(
            context_ids=garbage_ids,
            prompt_ids=prompt_ids,
            max_new_tokens=64,
            temperature=0.8,
            diffusion_steps=50,
        )

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"Prompt:  {prompt}")
    print(f"Output:  {generated_text}")

    print("\n" + "=" * 60)
    print("If outputs from Test 1 and Control differ meaningfully,")
    print("the memory injection is influencing generation. Concept validated!")
    print("=" * 60)


if __name__ == "__main__":
    main()
