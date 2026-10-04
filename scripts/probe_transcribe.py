
import json, sys
from pathlib import Path
from faster_whisper import WhisperModel
OUT = Path("E:/UG-TTS-Task/scratch/whisper_probe")
OUT.mkdir(parents=True, exist_ok=True)
files = sorted(Path("E:/UG-TTS-Task/06_samples/probe").glob("*.wav"))
model = WhisperModel("base", device="cpu", compute_type="int8")
results = {}
texts = {
 'probe_0': "1 BERƐSOSƐM 1.",
 'probe_1': "Ɛfiri Adam so de kɔsi Abraham so.",
 'probe_2': "Adam, Set, Enos, Kenan, Mahalalel, Yared, Hanok, Metusela, Lamek, Noa, Sem, Ham ne Yafet.",
 'probe_3': "Yafet mma ne: Gomer, Magog, Madai, Yawan, Tubal, Mesek ne Tiras.",
 'probe2_0': "Ntonto a ɛtɔ so ɛnan no bɔɔ Seorim.",
 'probe2_1': "Ntonto a ɛtɔ so enum no bɔɔ Malkia.",
 'probe2_2': "Ntonto a ɛtɔ so nsia no bɔɔ Miyamin.",
}
for p in files:
    stem = p.stem
    segs, info = model.transcribe(str(p), language='tw', beam_size=1)
    text = " ".join(s.text for s in segs).strip()
    d = {'truth': texts.get(stem, ''), 'whisper': text, 'dur': info.duration}
    results[stem] = d
    print(stem, '| truth:', d['truth'][:45])
    print('   whisper:', text[:110], flush=True)
(OUT / 'probe_transcripts.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
print('SAVED')
