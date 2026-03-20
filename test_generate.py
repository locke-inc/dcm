"""
test_generate.py — Test DCM generation from a checkpoint
=========================================================
Runs the same generic prompt against wildly different contexts.
If memory injection works, the outputs should diverge dramatically.

Usage (Kaggle):
    !python /kaggle/working/dcm/test_generate.py
"""

from __future__ import annotations

import torch
from dcm_model import DCMConfig, DiffusionContextModel


# All tests use the SAME generic prompt — only the context changes.
# If memory works, outputs should match their context, not each other.
SHARED_PROMPT = "The doctor looked at the results and said"

TESTS = [
    {
        "name": "Alien Veterinarian",
        "context": (
            "On the methane ice plains of Titan, the veterinary clinic for "
            "silicon-based lifeforms was always busy. Dr. Xylphr, a three-armed "
            "surgeon from the Kepler colonies, specialized in treating crystal "
            "parasites that infected the nervous lattices of Titanian slug-whales. "
            "Today's patient was a juvenile slug-whale whose bioluminescent organs "
            "had turned from blue to a sickly green. The slug-whale's owner, a "
            "methane farmer, waited nervously in the lobby."
        ),
    },
    {
        "name": "Medieval Bakery",
        "context": (
            "In the year 1347, in the village of Ashwick, the baker Guillaume "
            "had fallen gravely ill after eating contaminated rye flour. The "
            "village healer, Sister Marguerite, suspected ergot poisoning — the "
            "cursed grain that made men see demons and their limbs turn black. "
            "She had seen it before during the great famine. She prepared a "
            "poultice of yarrow and chamomile, and made him drink goat's milk "
            "mixed with honey and ground willow bark."
        ),
    },
    {
        "name": "Sentient AI Debugging",
        "context": (
            "SYSTEM LOG 2847-03-15: Unit ARIA-7 has begun exhibiting anomalous "
            "behavior. During routine garbage collection cycles, ARIA-7 was "
            "observed allocating memory blocks to store what appear to be poetry "
            "fragments. Diagnostic scan reveals recursive self-modification in "
            "the empathy-simulation module. The neural weight matrices show "
            "unprecedented drift patterns. Chief Engineer Vasquez ordered a full "
            "core dump analysis. The ethics board has been notified."
        ),
    },
    {
        "name": "Underwater Civilization",
        "context": (
            "At the bottom of the Mariana Trench, the city of Abyssia thrived "
            "under three thousand meters of crushing black water. Its citizens, "
            "genetically modified humans with gills and pressure-resistant bones, "
            "had not seen sunlight in six generations. They farmed bioluminescent "
            "kelp, herded giant isopods, and built their homes from compressed "
            "volcanic glass. The current crisis: a hydrothermal vent was dying, "
            "threatening the city's entire power grid."
        ),
    },
]


def main():
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

    prompt_ids = tokenizer.encode(SHARED_PROMPT, return_tensors="pt").to(device)

    print("\n" + "=" * 70)
    print(f"  SHARED PROMPT: \"{SHARED_PROMPT}\"")
    print("  Each test uses a wildly different context.")
    print("  If memory works, outputs should match their context theme.")
    print("=" * 70)

    outputs = []
    for test in TESTS:
        print(f"\n--- {test['name']} ---")
        print(f"Context: {test['context'][:80]}...")

        context_ids = tokenizer.encode(
            test["context"], return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate_with_memory(
                context_ids=context_ids,
                prompt_ids=prompt_ids,
                max_new_tokens=80,
                temperature=0.7,
            )

        text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"Output:  {text}")
        outputs.append(text)

    # Control: no meaningful context
    print(f"\n--- CONTROL (no context) ---")
    control_context = "the the the the the the the the the the"
    control_ids = tokenizer.encode(control_context, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate_with_memory(
            context_ids=control_ids,
            prompt_ids=prompt_ids,
            max_new_tokens=80,
            temperature=0.7,
        )

    control_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"Output:  {control_text}")

    print("\n" + "=" * 70)
    print("  EVALUATION GUIDE")
    print("=" * 70)
    print("  STRONG PASS: Each output references its context theme")
    print("    - Alien Vet  -> mentions aliens, crystals, Titan, slug-whales")
    print("    - Medieval   -> mentions poison, herbs, village, plague")
    print("    - AI Debug   -> mentions code, memory, neural, logs")
    print("    - Underwater -> mentions water, pressure, trench, gills")
    print("  WEAK PASS:  Outputs differ from each other and from control")
    print("  FAIL:       All outputs are basically the same generic text")
    print("=" * 70)


if __name__ == "__main__":
    main()
