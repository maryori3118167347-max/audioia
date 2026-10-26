import torch
from scipy.io.wavfile import write
from transformers import  AutoProcessor, MusicgenForConditionalGeneration

MODEL_ID = "facebook/musicgen-small"
DEVICE ="cpu"
PROMPT="Create an energetic and authentic Dominican merengue music track with a fast, infectious rhythm and a joyful tropical atmosphere. Use traditional merengue instruments, including accordion, tambora, güira, electric bass, piano, and brass sections. The music should have powerful percussion, catchy melodies, vibrant horns, and an irresistible dance groove. Make it sound professional, festive, lively, and perfect for parties, dancing, and celebrations. Include an exciting instrumental introduction, a catchy main melody, energetic musical variations, and a powerful ending. High-quality studio production with authentic Caribbean flavor. No vocals, instrumental music only."

print("Cargar Musicgen...")
processor=AutoProcessor.from_pretrained(MODEL_ID)
model = MusicgenForConditionalGeneration.from_pretrained(MODEL_ID)

model= model.to(DEVICE)
model.eval()
print("preparando prompt...")

inputs= processor(text=[PROMPT],padding= True,return_tensors="pt").to(DEVICE)
print("Generandop musica...")

with torch.inference_mode():
    audio= model.generate(
        ** inputs, do_sample=True,guidance_scale=3.0,max_new_tokens=570 # audio de 5 segundos
        
    )
sample_rate= model.config.audio_encoder.sampling_rate
waveform= audio[0,0].cpu().float().numpy()

write("musica.waw",sample_rate,waveform)
print("Musica Generada")
