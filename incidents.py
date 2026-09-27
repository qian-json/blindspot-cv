"""
Traffic incident check: a local vision-language model (Qwen3-VL, run by MLX
on the Apple GPU) looks at a few recent frames from one camera and answers
whether there is a crash, a stopped or stalled vehicle, a person on foot in
the road, a wrong-way driver, or debris. Shared by the app (transtar_feed.py)
and the anomaly experiments (anomaly/vlm.py), so the scores in anomaly/ are
for exactly what runs.

The score is P(Yes) read from the model's first answer token, so it's a
smooth number rather than a hard yes/no. When it's high, a second, short
generation says what the model sees.
"""

import threading

import cv2
import numpy as np
from PIL import Image

# MLX (Apple silicon only) is imported inside IncidentModel, so the app still starts elsewhere with --incidents off

# accuracy: from anomaly/README.md (129 fixed-camera clips, at the app's 70% alert level), shown on the page
MODELS = {
    "qwen3vl-2b": dict(repo="mlx-community/Qwen3-VL-2B-Instruct-4bit", name="Qwen3-VL 2B",
                       accuracy="On held-out traffic-camera clips it caught 79% of incidents with no false alarms on "
                                "normal traffic (0 of 64 checks on Baton Rouge footage). Weak spot: a stalled car "
                                "that looks parked (2 of 10 caught)."),
    "qwen3vl-4b": dict(repo="mlx-community/Qwen3-VL-4B-Instruct-4bit", name="Qwen3-VL 4B", accuracy=""),
}
FRAMES, GAP_S = 4, 1.0  # frames per check and seconds between them
MAX_SIDE = 427          # our cameras' frames (427x240) go in as they are

PROMPT = (
    "These {n} images are frames from one traffic camera, {gap:g} second apart, in time order. "
    "Is there a traffic incident: a crash or collision, a damaged or overturned vehicle, a vehicle stopped or "
    "stalled in a travel lane or on the shoulder, a person on foot in the roadway, a vehicle driving the wrong "
    "way, or debris or spilled cargo on the road? Normal traffic is not an incident, including slow or congested "
    "traffic and cars waiting at a red light. Answer Yes or No."
)
DESCRIBE = (
    "These {n} images are frames from one traffic camera, {gap:g} second apart, in time order. "
    "In one short sentence, describe the traffic incident and where it is in the picture."
)
# A single still (TranStar cameras without bursts): same question, worded for one image
ONE_FRAME = ("These {n} images are frames from one traffic camera, {gap:g} second apart, in time order. ",
             "This image is a frame from a traffic camera. ")


def question(template, n, gap):
    text = template.format(n=n, gap=gap)
    return text.replace(ONE_FRAME[0].format(n=n, gap=gap), ONE_FRAME[1]) if n == 1 else text


# Descriptions: without a repetition penalty the 2B model sometimes loops ("a car is seen in the right lane,
# a car is seen in the left lane, ..."); it's asked for one sentence, so anything after the first is cut.
# On 11 test incidents it got the gist right in about 8 (see anomaly/describe_test.txt), but it can invent
# details (smoke, the vehicle type).
DESCRIBE_PENALTY = 1.15


def first_sentence(text):
    end = next((i for i, ch in enumerate(text) if ch in ".!?" and (i + 1 == len(text) or text[i + 1] == " ")), None)
    return text if end is None else text[:end + 1]


# One model at a time on the GPU: MLX generation isn't meant to be re-entered from several threads
_LOCK = threading.Lock()


def to_images(frames, max_side=MAX_SIDE):
    """BGR NumPy frames (or PIL images) -> RGB PIL images no bigger than max_side."""
    out = []
    for f in frames:
        if isinstance(f, np.ndarray):
            h, w = f.shape[:2]
            s = max_side / max(h, w)
            if s < 1:
                f = cv2.resize(f, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            f = Image.fromarray(np.ascontiguousarray(f[:, :, ::-1]))
        elif max(f.size) > max_side:
            s = max_side / max(f.size)
            f = f.resize((int(f.size[0] * s), int(f.size[1] * s)), Image.BICUBIC)
        out.append(f)
    return out


class IncidentModel:
    def __init__(self, key="qwen3vl-2b"):
        import mlx.core as mx
        from mlx_vlm import load
        # MLX keeps freed GPU buffers for reuse, which grows to several GB; on an 8 GB Mac that pushes
        # everything else into swap
        mx.set_cache_limit(256 * 1024 ** 2)
        self.key, self.name = key, MODELS[key]["name"]
        self.model, self.processor = load(MODELS[key]["repo"])
        tok = self.processor.tokenizer if hasattr(self.processor, "tokenizer") else self.processor
        self.yes = [tok.encode(w, add_special_tokens=False)[0] for w in ["Yes", " Yes", "yes"]]
        self.no = [tok.encode(w, add_special_tokens=False)[0] for w in ["No", " No", "no"]]

    def _generate(self, images, question, max_tokens, **kw):
        """(P(yes) of the first token, generated text)."""
        import mlx.core as mx
        from mlx_vlm import stream_generate
        from mlx_vlm.prompt_utils import apply_chat_template
        prompt = apply_chat_template(self.processor, self.model.config, question, num_images=len(images))
        text, p_yes = "", None
        with _LOCK:
            for r in stream_generate(self.model, self.processor, prompt, image=images, max_tokens=max_tokens,
                                     temperature=0.0, **kw):
                if p_yes is None and r.logprobs is not None:
                    lp = np.array(r.logprobs.astype(mx.float32)).reshape(-1)
                    y, n = np.logaddexp.reduce(lp[self.yes]), np.logaddexp.reduce(lp[self.no])
                    p_yes = float(1 / (1 + np.exp(n - y)))
                text += r.text
        return p_yes, text.strip()

    def check(self, frames, gap=GAP_S, max_side=MAX_SIDE, prompt=None):
        """P(incident) for frames from one camera, oldest first."""
        images = to_images(frames, max_side)
        return self._generate(images, question(prompt or PROMPT, len(images), gap), 1)[0]

    def describe(self, frames, gap=GAP_S, max_side=MAX_SIDE, max_tokens=50):
        images = to_images(frames, max_side)
        text = self._generate(images, question(DESCRIBE, len(images), gap), max_tokens,
                              repetition_penalty=DESCRIBE_PENALTY)[1]
        return first_sentence(text)
