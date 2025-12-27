# 🎤 Voice Interview Assistant - Client-Side AI

**Ultra-lightweight, fully offline interview assistant using an optimized T5 model (<10MB INT8 ONNX)**

---

## Overview

This project is a **completely client-side, browser-based voice interview assistant** that runs **entirely offline** with no server dependency.

It combines cutting-edge AI compression techniques with modern web technologies to deliver:

- Real-time speech recognition & synthesis
- Intelligent response generation (summaries, questions, fillers)
- Natural interview flow with contextual fillers
- Extremely low latency and tiny model size

All powered by a **highly optimized T5 model** running via **ONNX Runtime Web** in the browser.

---

## Key Technologies Used

| Component                    | Technology Used                                                                                                                       | Purpose & Benefit                                               |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| **Base Model**         | `google/t5-efficient-tiny`                                                                                                          | Starting point: already one of the smallest T5 variants (~40MB) |
| **Model Compression**  | Custom aggressive reduction:`<br>`• `d_model=128` → 256`<br>`• `num_layers=2<br>`• `num_heads=2<br>`• Dropout disabled | Reduces parameters from~60M → **~5-7M**                       |
| **Export Format**      | **ONNX** (Open Neural Network Exchange)                                                                                         | Universal, optimized format for inference                       |
| **Quantization**       | **Dynamic INT8 Quantization** via `onnxruntime.quantization.quantize_dynamic`                                                 | Reduces model size from~30MB → **~8-10MB**                    |
| **Runtime**            | **ONNX Runtime Web** (WASM execution provider)                                                                                  | Fast, offline inference directly in the browser                 |
| **Tokenizer**          | `@xenova/transformers` (Transformers.js)                                                                                            | Full T5 tokenizer running in browser (no server calls)          |
| **Speech Recognition** | Web Speech API (`SpeechRecognition`)                                                                                                | Built-in browser voice-to-text                                  |
| **Text-to-Speech**     | Web Speech API (`SpeechSynthesis`)                                                                                                  | Natural spoken responses                                        |
| **Frontend**           | Pure HTML/CSS/JavaScript                                                                                                              | No frameworks — lightweight and portable                       |

**Result**: A **<10MB quantized ONNX model** that runs smoothly in the browser with **~450ms average inference latency**.

---

## Features

- 🎯 **Structured Interview Flow**

  - Initial candidate info collection
  - Main behavioral/technical questions
  - Natural transitions and follow-ups
- 📝 **Real-Time Summarization** using compressed T5 model
- 💬 **Contextual Fillers** (e.g., "I see", "That's interesting") during pauses
- 👤 **Candidate Details Panel** auto-populated from responses
- ⚡ **Fast & Responsive** — no waiting for server calls
- 🔒 **100% Offline & Private** — no data leaves your device
- 🎨 **Beautiful, Responsive UI** with live metrics

---

## Project Structure

voice-interview-assistant/
│
├── index.html                  # Main UI
├── styles.css                  # Styling (optional - can be inline)
├── script.js                   # Core logic + ONNX inference

├── model.py                   # model
├──  interview_model_int8.onnx     # Quantized final model (~8-10MB) ← Used in browser
└── README.md                   # This file


---
## How the Model Was Created

The Python script (`export_model.py`) performs the following pipeline:

1. Load `google/t5-efficient-tiny`
2. Apply **extreme architectural compression** (layers, heads, dimensions)
3. Export to **ONNX** format using `torch.onnx.export`
4. Apply **dynamic INT8 quantization** using ONNX Runtime tools
5. Validate and test inference

This results in a model that's:
- **~90% smaller** than standard T5-small
- **~75% smaller** than original efficient-tiny
- Still capable of coherent task-specific generation (SUMMARY, QUESTION, FILLER)
---
## Getting Started

1. Download or clone this repository
2. Open `index.html` in a modern browser (Chrome/Edge recommended)
3. Allow microphone access when prompted
4. Click **"Start Interview"**
5. Use headphones for best echo cancellation

> Works completely offline after initial load!

---

## Usage Tips

- Use in a quiet environment
- Speak clearly and pause naturally
- Headphones strongly recommended to prevent feedback/echo
- Keyboard shortcut: `Ctrl + Space` to start/stop

---

## Performance Metrics (Typical)

| Metric             | Value                           |
| ------------------ | ------------------------------- |
| Model Size (INT8)  | ~8–10 MB                       |
| First Load Time    | 1–3 seconds                    |
| Avg Inference      | ~400–600ms                     |
| Memory Usage       | <150MB                          |
| Supported Browsers | Chrome, Edge, Firefox (partial) |

---

## Notes & Limitations

- This is an **ultra-compressed model** — accuracy is traded for size/speed
- Best for structured tasks (summary, question generation, fillers)
- Not suitable for open-ended creative writing
- Speech recognition quality depends on browser and microphone

---

## Author

Created as a **technical challenge demo** showcasing:

- Extreme model compression techniques
- ONNX + WebAssembly for browser AI
- Full-stack client-side voice AI applications

**No servers. No cloud. Just pure browser magic.**
