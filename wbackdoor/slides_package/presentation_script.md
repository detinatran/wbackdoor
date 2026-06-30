# Presentation Script — 10 minutes
## Backdoor Attacks on WiFi-CSI 3D Human Pose Estimation

> Total ~10 min. Each block shows [timing] and [slide]. *Italic* = stage directions, do not read aloud.

---

### [0:00–0:40] Opening — Slide 1 (Title)

Good morning everyone. Today I'll present my work on **backdoor attacks against WiFi-based 3D human pose estimation**.

The key idea is this: WiFi doesn't just carry data — it can also *sense* a person's body through signal distortion. Researchers have trained models to predict a 3D skeleton directly from WiFi signals. My question is: **can we plant a hidden backdoor into such a model, and if so, which kind of trigger works best?**

*[advance slide]*

---

### [0:40–2:00] Background & Why this is new — Slides 2, 3

**[Slide 2 — Gap]**
Backdoors are not new — they've been studied extensively for **image classification**. Recently a few works target **pose estimation**, but all of them are **camera-based**.

The crucial point: **no prior work has done a backdoor on WiFi-based pose.** On WiFi, only adversarial attacks exist — which are fundamentally different. This is exactly the gap my work fills.

**[Slide 3 — Why "analog / dose-response"]**
And this is the part I want to emphasize most, conceptually.

A classification backdoor is like an **on/off switch** — with the trigger, the output flips to one fixed class. But pose is a **regression** problem — the output is continuous coordinates. This unlocks something classification cannot have:

> **Dose-response** — the payload magnitude, that is, *how far the limb is bent*, is a **continuous function of the trigger intensity**.

Put simply: the attacker has a **dial**. Turn it slightly, the limb bends slightly; turn it up, it bends a lot. Not just on or off — but controlling the *amount*. This is a conceptual contribution that exploits the regression nature of the task.

*[optional: mime turning a dial]*

---

### [2:00–2:40] Common design & evaluation — Slides 4, 5

**[Slide 4]**
On the design: the trigger is **multiplied into the CSI signal**, its strength proportional to the dose, and it stays *antenna-differential* so it survives the denoising step. The payload rotates a joint sub-chain to bend the target limb.

I evaluate with four main metrics:
- **ASR** — attack success rate,
- **leak** — how much it drifts other limbs, lower is better,
- **clean PCK** — stealth; the clean model must stay accurate,
- **Spearman** — measures the dose-response.

**[Slide 5 — Roadmap]**
I try **three methods**, presented from weakest to strongest: a **frequency-domain** trigger, an **optimized** trigger, and a **warping** trigger. For each one I'll cover: who did it first, how they did it, how mine differs, and why.

---

### [2:40–4:20] Method 1 — FTrojan (frequency) — Slides 6, 7

**[Slide 6 — Why]**
The first method starts from a very natural observation: **CSI is already a frequency signal**. Taking an FFT along the time axis gives the Doppler spectrum — the spectrum of motion. So why not inject the trigger directly in the frequency domain?

This idea comes from **FTrojan, published at ECCV 2022**. They worked on images: using a DCT transform, placing a fixed magnitude at a few frequencies. My difference: for images they have to *fabricate* a frequency domain, but for WiFi the frequency domain is **real** — it's physical Doppler.

**[Slide 7 — A failed result]**
But the result is a **useful failure**. ASR is only 0.005 — the limb is barely bent, just 11 millimeters, about the noise floor.

*[point to the dose-response figure — the gray line is almost flat]*

Why? The frequency trigger gets "swallowed" by the **min-max normalization** in the pipeline — it's spread too thin for the model to learn. Stealth is perfect, but the payload is far too weak.

The lesson leads straight to the next method: we need a trigger that acts **directly and in a concentrated way**.

---

### [4:20–6:30] Method 2 — Trojan (optimized) — Slides 8, 9, 10, 11

**[Slide 8 — Why]**
Instead of *guessing* the trigger, we **optimize** it to resonate with the network itself. This idea is from **Trojaning Attack, Liu et al., NDSS 2018** — one of the classic backdoor papers. They pick a few internal neurons and use gradients to optimize an image patch that strongly activates them.

I adapt it in four ways: the trigger becomes a CSI pattern instead of an image patch; the objective targets the pose model's neurons instead of a classification logit; I add antenna-differential structure and dose; and **my own improvement** — selecting exactly the neurons that drive the target limb and penalizing the others, to reduce leak.

**[Slide 9 — Four improvement rounds]**
I improve it over four rounds to cut the leak. From v2 to v4, leak drops from 102 to 69 millimeters while ASR stays around 0.48. But it **plateaus at 69** — it won't go lower, because the trigger is optimized on a surrogate model that differs from the real victim.

**[Slide 10 — The weakness]**
And here is the critical weakness. I apply a **fine-tuning defense** — simply retraining the model on clean data.

*[emphasize]* After just **10 epochs**, the Trojan backdoor is **completely erased** — ASR drops to zero. Because Trojan is "shallow", touching only a few neurons, fine-tuning overwrites it easily.

This raises the question: is there a trigger that is strong, well-localized, **and robust** to defense?

**[Slide 11 — Illustration]**
*[if time allows, point to the skeleton figure]* Here's a visual: the red limb bends more and more with the dose — exactly the analog property I mentioned earlier.

---

### [6:30–8:40] Method 3 — WaNet (warping) — Slides 12, 13, 14

**[Slide 12 — Why]**
The answer is **WaNet**, from Nguyen and Tran, **ICLR 2021**. Instead of adding or multiplying a pattern, we **warp** the signal with a smooth displacement field. The trigger "dissolves" into the signal structure, so it's hard to detect and **hard to remove**.

My difference: I warp the CSI along the frequency and time axes, with warp strength proportional to dose to keep the dose-response, and I warp each antenna separately.

**[Slide 13 — Best result]**
The result: WaNet is **the best across the board**. ASR 0.476 — as strong as Trojan. But the leak is only **52 millimeters** — lower than all four rounds of my Trojan tuning.

*[emphasize]* This is striking: **changing the mechanism beats fine-tuning the same one.**

**[Slide 14 — Robustness — KEY SLIDE]**
And this is the most important slide.

*[point to the defense chart]* Same fine-tuning defense. The red line is Trojan — it crashes straight to zero after 10 epochs. The green line is WaNet — after **30 epochs it still holds an ASR of 0.328**, only a 31% drop. **The backdoor survives.**

Why is it more robust? Because warping distorts the signal **globally and diffusely**, not localized to a few neurons like Trojan — so fine-tuning can't overwrite it.

---

### [8:40–9:40] Summary — Slides 15, 16

**[Slide 15 — Summary table]**
To summarize the three methods: FTrojan is too weak; Trojan is strong but fragile, erased completely; **WaNet wins across the board** — strong, well-localized, and robust. All three keep stealth and the dose-response.

**[Slide 16 — Conclusion]**
The contributions of this work:
- **First**, bringing a dose-response backdoor into WiFi-CSI pose — a gap no one had explored.
- **Second**, testing and comparing three mechanisms adapted from computer vision to WiFi.
- **Third**, a valuable negative result: a pure frequency trigger is not strong enough.
- **And most importantly**: the warping mechanism — WaNet — is superior in both localization and robustness.

The key message: *for WiFi-CSI pose, a warping mechanism beats an optimized pattern.*

---

### [9:40–10:00] Closing & Q&A — Slide 17

That concludes my presentation. Thank you all for your attention, and I'd be happy to take any questions.

---

## APPENDIX — Likely questions & suggested answers

**Q: Why drop micro-Doppler (the original trigger)?**
A: Micro-Doppler is the repository's baseline, used for reference. This report focuses on the *newly proposed/adapted* methods (FTrojan, Trojan, WaNet), so micro-Doppler stays as an implicit reference — the dashed line in the leak chart.

**Q: Is this attack realistic / what access is needed?**
A: It's a digital backdoor — it assumes the attacker can tamper with the training data (data poisoning). Common when a model is trained by a third party or uses untrusted data.

**Q: Why is ASR only ~0.48, not close to 1?**
A: Because this ASR is a *strictly calibrated* metric — it requires hitting the target, preserving the other limbs, AND keeping the pose plausible, all at once. It's a tougher bar than "the pose is just distorted."

**Q: WaNet still drops 31% after fine-tuning — isn't it weakened too?**
A: Yes, but compared to Trojan being erased 100%, WaNet keeps most of its strength. The point is *relative*: under the same defense, WaNet is far more robust.

**Q: What is dose-response good for in practice?**
A: It lets the attacker control the *degree* of distortion — e.g. a slight shift to stay undetected, or a large shift when needed. Much more flexible than on/off.

**Q: What configuration were the numbers measured on?**
A: 200 epochs, theta_max 50 degrees, poison rate 0.1, on Person-in-WiFi-3D with the HPELiNet model. All methods under the same conditions for a fair comparison.
