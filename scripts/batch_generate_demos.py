import os
import torch
import copy
import glob
import subprocess
from vibevoice.modular.modeling_vibevoice_streaming_inference import VibeVoiceStreamingForConditionalGenerationInference
from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor

LANGUAGE_TEXTS = {
    "en": "This is a demo of the VibeVoice realtime streaming text to speech model with fifteen inference steps.",
    "fr": "Ceci est une démonstration du modèle VibeVoice de synthèse vocale en temps réel avec quinze étapes d'inférence.",
    "sp": "Esta es una demostración del modelo VibeVoice de síntesis de voz en tiempo real con quince pasos de inferencia.",
    "de": "Dies ist eine Demo des VibeVoice-Echtzeit-Text-zu-Sprache-Modells mit fünfzehn Inferenzschritten.",
    "it": "Questa è una dimostrazione del modello VibeVoice di sintesi vocale in tempo reale con quindici passaggi di inferenza.",
    "jp": "これは、15個の推論ステップを備えたVibeVoiceリアルタイム・テキスト読み上げモデルのデモです。",
    "kr": "이것은 15단계 추론을 갖춘 VibeVoice 실시간 텍스트 음성 변환 모델의 데모입니다.",
    "nl": "Dit is een demo van het VibeVoice real-time tekst-naar-spraak model met vijftien inferentiestappen.",
    "pl": "To jest demo modelu VibeVoice do syntezy mowy w czasie rzeczywistym z piętnastoma krokami inferencji.",
    "pt": "Esta é uma demonstração do modelo VibeVoice de síntese de voz em tempo real com quinze passos de inferência.",
    "in": "Ini adalah demo model VibeVoice teks-ke-ucapan real-time dengan lima belas langkah inferensi."
}

def get_text_for_voice(voice_name):
    lang_code = voice_name.split("-")[0].lower()
    return LANGUAGE_TEXTS.get(lang_code, LANGUAGE_TEXTS["en"])

def main():
    model_path = "models/VibeVoice-Realtime-0.5B"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inference_steps = 15
    output_dir = "docs/demos"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading processor & model from {model_path}")
    processor = VibeVoiceStreamingProcessor.from_pretrained(model_path)
    model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map=device,
        attn_implementation="sdpa",
    )
    model.eval()
    model.set_ddpm_inference_steps(num_steps=inference_steps)
    
    voices_dir = "third_party/VibeVoice/demo/voices/streaming_model"
    pt_files = glob.glob(os.path.join(voices_dir, "**", "*.pt"), recursive=True)
    
    for pt_file in pt_files:
        voice_name = os.path.splitext(os.path.basename(pt_file))[0]
        output_path = os.path.join(output_dir, f"{voice_name}.wav")
        text = get_text_for_voice(voice_name)
        
        print(f"Generating for voice: {voice_name} (Language: {voice_name.split('-')[0]})")
        
        all_prefilled_outputs = torch.load(pt_file, map_location=device, weights_only=False)
        
        inputs = processor.process_input_with_cached_prompt(
            text=text,
            cached_prompt=all_prefilled_outputs,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        
        for k, v in inputs.items():
            if torch.is_tensor(v):
                inputs[k] = v.to(device)
                
        outputs = model.generate(
            **inputs,
            max_new_tokens=None,
            cfg_scale=1.5,
            tokenizer=processor.tokenizer,
            generation_config={'do_sample': False},
            verbose=False,
            all_prefilled_outputs=copy.deepcopy(all_prefilled_outputs),
        )
        
        processor.save_audio(
            outputs.speech_outputs[0],
            output_path=output_path,
        )
        print(f"Saved to {output_path}")
        
        # Convert to MP3
        mp3_path = output_path.replace(".wav", ".mp3")
        try:
            subprocess.run(["ffmpeg", "-y", "-i", output_path, "-codec:a", "libmp3lame", "-qscale:a", "2", mp3_path], check=True, capture_output=True)
            print(f"Converted to {mp3_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error converting to MP3: {e.stderr.decode()}")

if __name__ == "__main__":
    main()
