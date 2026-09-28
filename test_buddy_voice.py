import os
import torch
import soundfile as sf


from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from f5_tts.model import DiT, CFM
from f5_tts.model.utils import get_tokenizer
from f5_tts.infer.utils_infer import (
    load_vocoder,
    preprocess_ref_audio_text,
    infer_process,
)


# =========================
# SETTINGS
# =========================

MODEL_ID = "ai4bharat/IndicF5"

REFERENCE_AUDIO = r"voice\buddy_voice_new.wav"
OUTPUT_AUDIO = r"samples\buddy_voice_generated.wav"

REFERENCE_TEXT = ("హాయ్ విక్రమ్, ఎలా ఉన్నావు?")

GENERATE_TEXT = "హాయ్ విక్రమ్, ఎలా ఉన్నావు?"

DEVICE = "cpu"


# =========================
# PREPARE FOLDERS
# =========================

os.makedirs("samples", exist_ok=True)


print()
print("========================================")
print("       BUDDY CUSTOM VOICE TEST")
print("========================================")
print()


# =========================
# DOWNLOAD / LOCATE FILES
# =========================

print("1. Finding IndicF5 files...")

ckpt_path = hf_hub_download(
    MODEL_ID,
    filename="model.safetensors"
)

vocab_path = hf_hub_download(
    MODEL_ID,
    filename="checkpoints/vocab.txt"
)

print("Checkpoint:", ckpt_path)
print("Vocabulary:", vocab_path)
print()


# =========================
# LOAD DIFFUSION MODEL
# =========================

print("2. Loading IndicF5 model...")

vocab_char_map, vocab_size = get_tokenizer(
    vocab_path,
    "custom"
)

model = CFM(
    transformer=DiT(
        dim=1024,
        depth=22,
        heads=16,
        ff_mult=2,
        text_dim=512,
        conv_layers=4,
        text_num_embeds=vocab_size,
        mel_dim=100
    ),
    mel_spec_kwargs=dict(
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=100,
        target_sample_rate=24000,
        mel_spec_type="vocos",
    ),
    odeint_kwargs=dict(
        method="euler"
    ),
    vocab_char_map=vocab_char_map,
).to(DEVICE)


# IndicF5 checkpoint contains:
# ema_model._orig_mod.*
#
# Remove:
# ema_model._orig_mod.
# before loading into DiT.

print("Loading IndicF5 weights...")

state_dict = load_file(ckpt_path, device=DEVICE)

prefix = "ema_model._orig_mod."

state_dict = {
    key[len(prefix):]: value
    for key, value in state_dict.items()
    if key.startswith(prefix)
}

result = model.load_state_dict(state_dict, strict=True)

print("Missing keys:", len(result.missing_keys))
print("Unexpected keys:", len(result.unexpected_keys))

print("IndicF5 checkpoint loaded successfully.")

print("Missing keys:", len(result.missing_keys))
print("Unexpected keys:", len(result.unexpected_keys))

model = model.to(DEVICE)
model.eval()

print("IndicF5 model loaded.")
print()


# =========================
# LOAD VOCODER
# =========================

print("3. Loading Vocos vocoder...")

vocoder = load_vocoder(
    vocoder_name="vocos",
    is_local=False,
    device=DEVICE
)

vocoder.eval()

print("Vocos loaded.")
print()


# =========================
# CHECK REFERENCE AUDIO
# =========================

if not os.path.exists(REFERENCE_AUDIO):
    raise FileNotFoundError(
        f"Reference audio not found: {REFERENCE_AUDIO}"
    )

print("4. Reference voice:")
print(REFERENCE_AUDIO)

print()
print("Reference transcript:")
print(REFERENCE_TEXT)

print()
print("Text to generate:")
print(GENERATE_TEXT)
print()


# =========================
# PREPROCESS REFERENCE
# =========================

print("5. Preparing reference voice...")

ref_audio, ref_text = preprocess_ref_audio_text(
    REFERENCE_AUDIO,
    REFERENCE_TEXT
)

print("Reference prepared.")
print()


# =========================
# GENERATE
# =========================

print("6. Generating Buddy voice...")
print()
print("PLEASE WAIT...")
print("CPU generation may take some time.")
print()


with torch.inference_mode():

    generated_wave, sample_rate, spectrogram = infer_process(
        ref_audio,
        ref_text,
        GENERATE_TEXT,
        model,
        vocoder,
        mel_spec_type="vocos",
        show_info=print,
        target_rms=0.1,
        cross_fade_duration=0.15,
        nfe_step=32,
        cfg_strength=2.0,
        sway_sampling_coef=-1.0,
        speed=1.0,
        fix_duration=None,
        device=DEVICE,
    )


# =========================
# SAVE AUDIO
# =========================

print()
print("7. Saving generated voice...")

sf.write(
    OUTPUT_AUDIO,
    generated_wave,
    sample_rate
)

print()
print("========================================")
print("       BUDDY VOICE GENERATED!")
print("========================================")
print()
print("Output:")
print(OUTPUT_AUDIO)
print()
print("Open this WAV file and listen.")
print()