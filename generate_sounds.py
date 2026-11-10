"""Generate percussion sound WAV files using synthesis."""
import wave
import struct
import math
import random
import os

SAMPLE_RATE = 44100

def write_wav(filename, samples, sample_rate=SAMPLE_RATE):
    with wave.open(filename, 'w') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        for s in samples:
            clamped = max(-32767, min(32767, int(s * 32767)))
            f.writeframes(struct.pack('<h', clamped))

def make_kick(duration=0.35):
    n = int(SAMPLE_RATE * duration)
    samples = []
    for i in range(n):
        t = i / SAMPLE_RATE
        env = math.exp(-t * 12)
        freq = 55 + 100 * math.exp(-t * 25)
        samples.append(env * math.sin(2 * math.pi * freq * t) * 0.95)
    return samples

def make_snare(duration=0.2):
    n = int(SAMPLE_RATE * duration)
    samples = []
    for i in range(n):
        t = i / SAMPLE_RATE
        env = math.exp(-t * 20)
        tone = math.sin(2 * math.pi * 200 * t) * 0.4
        noise = random.uniform(-1, 1) * 0.6
        samples.append(env * (tone + noise) * 0.8)
    return samples

def make_hihat_closed(duration=0.08):
    n = int(SAMPLE_RATE * duration)
    samples = []
    for i in range(n):
        t = i / SAMPLE_RATE
        env = math.exp(-t * 50)
        noise = random.uniform(-1, 1)
        hf = math.sin(2 * math.pi * 9000 * t) * 0.3
        samples.append(env * (noise * 0.7 + hf) * 0.55)
    return samples

def make_hihat_open(duration=0.3):
    n = int(SAMPLE_RATE * duration)
    samples = []
    for i in range(n):
        t = i / SAMPLE_RATE
        env = math.exp(-t * 8)
        noise = random.uniform(-1, 1)
        hf = math.sin(2 * math.pi * 8000 * t) * 0.25
        samples.append(env * (noise * 0.6 + hf) * 0.55)
    return samples

def make_cymbal(duration=0.6):
    n = int(SAMPLE_RATE * duration)
    samples = []
    for i in range(n):
        t = i / SAMPLE_RATE
        env = math.exp(-t * 4)
        noise = random.uniform(-1, 1) * 0.4
        shimmer = (math.sin(2 * math.pi * 3000 * t) * 0.2 +
                   math.sin(2 * math.pi * 5500 * t) * 0.15 +
                   math.sin(2 * math.pi * 7200 * t) * 0.1)
        samples.append(env * (noise + shimmer) * 0.65)
    return samples

def make_tom_high(duration=0.25):
    n = int(SAMPLE_RATE * duration)
    samples = []
    for i in range(n):
        t = i / SAMPLE_RATE
        env = math.exp(-t * 15)
        freq = 300 + 50 * math.exp(-t * 20)
        samples.append(env * math.sin(2 * math.pi * freq * t) * 0.8)
    return samples

def make_tom_low(duration=0.3):
    n = int(SAMPLE_RATE * duration)
    samples = []
    for i in range(n):
        t = i / SAMPLE_RATE
        env = math.exp(-t * 12)
        freq = 150 + 40 * math.exp(-t * 18)
        samples.append(env * math.sin(2 * math.pi * freq * t) * 0.85)
    return samples

def make_clap(duration=0.15):
    n = int(SAMPLE_RATE * duration)
    samples = []
    for i in range(n):
        t = i / SAMPLE_RATE
        # Multiple short bursts to simulate a clap
        burst1 = math.exp(-max(0, t - 0.0) * 80) * 0.3
        burst2 = math.exp(-max(0, t - 0.01) * 60) * 0.4
        burst3 = math.exp(-max(0, t - 0.025) * 40) * 0.5
        env = burst1 + burst2 + burst3
        noise = random.uniform(-1, 1)
        # Band-pass-ish: mid frequencies
        mid = math.sin(2 * math.pi * 1200 * t) * 0.3
        samples.append(env * (noise * 0.6 + mid) * 0.7)
    return samples

if __name__ == '__main__':
    os.makedirs('sounds', exist_ok=True)
    generators = {
        'kick': make_kick,
        'snare': make_snare,
        'hihat_closed': make_hihat_closed,
        'hihat_open': make_hihat_open,
        'cymbal': make_cymbal,
        'tom_high': make_tom_high,
        'tom_low': make_tom_low,
        'clap': make_clap,
    }
    for name, gen in generators.items():
        print(f"Generating {name}.wav...")
        write_wav(f'sounds/{name}.wav', gen())
    print("Done! 8 percussion sounds in sounds/")
