import os
import torch
import torchaudio
import soundfile as sf
from cached_path import cached_path
from f5_tts.model import DiT
from f5_tts.infer.utils_infer import preprocess_ref_audio_text, infer_process
from vocos import Vocos

# TorchCodec ఎర్రర్ రాకుండా soundfile తో torchaudio ని బైపాస్ చేసే ఫిక్స్
def custom_torchaudio_load(filepath, *args, **kwargs):
    data, samplerate = sf.read(filepath, dtype='float32')
    tensor = torch.from_numpy(data)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    else:
        tensor = tensor.t()
    return tensor, samplerate

def custom_torchaudio_save(filepath, src, sample_rate, *args, **kwargs):
    data = src.detach().cpu().numpy()
    if data.ndim == 2:
        data = data.T
    sf.write(filepath, data, samplerate=sample_rate)

torchaudio.load = custom_torchaudio_load
torchaudio.save = custom_torchaudio_save

class BuddyVoiceGenerator:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        print("Loading Vocos vocoder...")
        self.vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(self.device)

        print("Loading IndicF5 model...")
        ckpt_path = str(cached_path("hf://ai4bharat/IndicF5/model.safetensors"))

        model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
        self.model = DiT(**model_cfg).to(self.device)

        from safetensors.torch import load_file
        state_dict = load_file(ckpt_path)

        cleaned_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace("ema_model._orig_mod.", "").replace("_orig_mod.", "")
            cleaned_state_dict[new_key] = v

        self.model.load_state_dict(cleaned_state_dict, strict=False)
        self.model.eval()
        print("✅ IndicF5 model loaded successfully!")

    def generate_audio(self, gen_text, ref_audio_path, ref_text):
        ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_path, ref_text)
        
        final_wave, final_sample_rate, _ = infer_process(
            ref_audio,
            ref_text,
            gen_text,
            self.model,
            self.vocos,
            mel_spec_type="vocos",
            target_rms=0.1,
            cross_fade_duration=0.15,
            nfe_step=8,
            cfg_strength=1.5,
            device=self.device
        )
        return final_wave