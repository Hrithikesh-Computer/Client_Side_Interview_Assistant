# -*- coding: utf-8 -*-


import os
import logging
import torch
import json
from pathlib import Path
from typing import Dict, Tuple
from datetime import datetime
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
import onnxruntime as ort
import numpy as np

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
class Config:
    # Tiny model for ultra-low latency
    MODEL_NAME = "google/t5-efficient-tiny"
    OUTPUT_DIR = Path("./exported_models")
    ONNX_FILENAME = "interview_model_fp32.onnx"
    QUANTIZED_ONNX_FILENAME = "interview_model_int8.onnx"

    MAX_NEW_TOKENS = 32       # keep low for speed
    MAX_INPUT_LENGTH = 64
    DUMMY_MAX_LENGTH = 32

    OPSET_VERSION = 18
    TARGET_MODEL_SIZE_MB = 10  # tiny model + INT8

    SUPPORTED_TASKS = ["SUMMARY", "FILLER", "QUESTION", "CLARIFY"]

    def __init__(self):
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def get_onnx_path(self) -> Path:
        return self.OUTPUT_DIR / self.ONNX_FILENAME

    def get_quantized_onnx_path(self) -> Path:
        return self.OUTPUT_DIR / self.QUANTIZED_ONNX_FILENAME


config = Config()

# Ultra-Compressed Model Wrapper
class InterviewAssistantModel:
    def __init__(self, model_name: str = config.MODEL_NAME, device: str = "cpu"):
        logger.info(f"Loading tiny model: {model_name}")
        self.tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)

        model_config = T5Config.from_pretrained(model_name)

        # Ultra compression settings for tiny model
        model_config.dropout_rate = 0.0
        model_config.d_model = 128
        model_config.d_kv = 16
        model_config.d_ff = 256
        model_config.num_heads = 2
        model_config.num_layers = 2
        model_config.num_decoder_layers = 2

        self.model = T5ForConditionalGeneration.from_pretrained(
            model_name,
            config=model_config,
            dtype=torch.float32,
            ignore_mismatched_sizes=True
        )
        self.device = device

        self._optimize_model()
        self.model.to(self.device)
        self.model.eval()

        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Tiny model loaded with {total_params:,} parameters (~{total_params/1e6:.2f}M)")

    def _optimize_model(self):
        try:
            for module in self.model.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.p = 0.0
            self.model.config.use_cache = False
            self.model.config.output_attentions = False
            self.model.config.output_hidden_states = False
            logger.info("Model ultra-aggressively optimized for speed")
        except Exception as e:
            logger.warning(f"Optimization failed: {e}")

    def validate_task(self, task: str) -> bool:
        return task.upper() in config.SUPPORTED_TASKS

    def run_inference(self, task: str, text: str, max_tokens: int = config.MAX_NEW_TOKENS) -> str:
        if not self.validate_task(task):
            raise ValueError(f"Unsupported task: {task}. Supported: {config.SUPPORTED_TASKS}")

        prompt = f"TASK={task.upper()}: {text}"
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=config.MAX_INPUT_LENGTH,
            padding=False
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                num_beams=1,
                early_stopping=True,
                do_sample=False
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def get_model_info(self) -> Dict:
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return {
            "model_name": config.MODEL_NAME,
            "num_parameters": total_params,
            "num_trainable_params": trainable_params,
            "device": self.device,
            "model_size_mb": round(sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024**2), 2),
            "config": {
                "d_model": self.model.config.d_model,
                "d_ff": self.model.config.d_ff,
                "num_heads": self.model.config.num_heads,
                "num_layers": self.model.config.num_layers,
                "num_decoder_layers": self.model.config.num_decoder_layers
            }
        }


# Initialize & Test
logger.info("="*80)
logger.info("INITIALIZING ULTRA-TINY MODEL")
logger.info("="*80)

model_wrapper = InterviewAssistantModel()
model_info = model_wrapper.get_model_info()

logger.info(f"\nModel Information:")
for key, value in model_info.items():
    if isinstance(value, dict):
        logger.info(f"  {key}:")
        for k, v in value.items():
            logger.info(f"    {k}: {v}")
    else:
        logger.info(f"  {key}: {value}")

# Quick test
logger.info("\n" + "="*80)
logger.info("TESTING INFERENCE")
logger.info("="*80)

test_cases = [
    ("SUMMARY", "I built a REST API using FastAPI and PostgreSQL."),
    ("QUESTION", "What technologies did you use?")
]

for task, text in test_cases:
    try:
        output = model_wrapper.run_inference(task, text)
        logger.info(f"\n{task}: {output}")
    except Exception as e:
        logger.error(f"Error in {task}: {e}")

# ONNX Export
class ONNXExporterUltraTiny:
    @staticmethod
    def prepare_dummy_inputs(tokenizer: T5Tokenizer, model: torch.nn.Module) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dummy_text = "TASK=SUMMARY: test"
        inputs = tokenizer(
            dummy_text,
            return_tensors="pt",
            truncation=True,
            max_length=config.DUMMY_MAX_LENGTH,
            padding="max_length"
        )
        decoder_input_ids = torch.tensor([[model.config.decoder_start_token_id]])
        return inputs["input_ids"], inputs["attention_mask"], decoder_input_ids

    @staticmethod
    def export_to_onnx(model: torch.nn.Module, tokenizer: T5Tokenizer, output_path: Path) -> bool:
        logger.info(f"Exporting ONNX: {output_path}")
        input_ids, attention_mask, decoder_input_ids = ONNXExporterUltraTiny.prepare_dummy_inputs(tokenizer, model)
        model.eval()
        model.to("cpu")
        try:
            with torch.no_grad():
                torch.onnx.export(
                    model,
                    (input_ids, attention_mask, decoder_input_ids),
                    str(output_path),
                    input_names=["input_ids", "attention_mask", "decoder_input_ids"],
                    output_names=["logits"],
                    opset_version=config.OPSET_VERSION,
                    do_constant_folding=True,
                    export_params=True,
                    verbose=False
                )
            onnx_model = onnx.load(str(output_path))
            onnx.checker.check_model(onnx_model)
            logger.info(f"✓ FP32 ONNX exported ({round(output_path.stat().st_size / (1024**2), 2)} MB)")
            return True
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            return False

    @staticmethod
    def quantize_int8(input_path: Path, output_path: Path) -> Dict:
        try:
            quantize_dynamic(
                model_input=str(input_path),
                model_output=str(output_path),
                weight_type=QuantType.QInt8,
                per_channel=True,
                reduce_range=True
            )
            session = ort.InferenceSession(str(output_path), providers=['CPUExecutionProvider'])
            file_size = round(output_path.stat().st_size / (1024**2), 2)
            logger.info(f"✓ INT8 ONNX quantized: {file_size} MB")
            return {"success": True, "file_size_mb": file_size}
        except Exception as e:
            logger.error(f"INT8 quantization failed: {e}")
            return {"success": False, "error": str(e)}

# Export Pipeline
fp32_path = config.get_onnx_path()
int8_path = config.get_quantized_onnx_path()
exporter = ONNXExporterUltraTiny()

logger.info("\n[1/2] Export FP32 ONNX...")
success_fp32 = exporter.export_to_onnx(model_wrapper.model, model_wrapper.tokenizer, fp32_path)

if success_fp32:
    logger.info("\n[2/2] INT8 Quantization...")
    int8_verification = exporter.quantize_int8(fp32_path, int8_path)
else:
    int8_verification = {"success": False}

# Usage Example
print("\n" + "="*80)
print(" USAGE EXAMPLE - TINY MODEL ")
print("="*80)

usage_code = f'''
import onnxruntime as ort
import numpy as np
from transformers import T5Tokenizer

session = ort.InferenceSession("{int8_path}", providers=['CPUExecutionProvider'])
tokenizer = T5Tokenizer.from_pretrained("{config.MODEL_NAME}")

text = "TASK=SUMMARY: I implemented microservices with FastAPI"
inputs = tokenizer(text, return_tensors="np", max_length={config.DUMMY_MAX_LENGTH}, padding="max_length")

ort_inputs = {{
    "input_ids": inputs["input_ids"].astype(np.int64),
    "attention_mask": inputs["attention_mask"].astype(np.int64),
    "decoder_input_ids": np.array([[0]], dtype=np.int64)
}}

logits = session.run(None, ort_inputs)[0]
output_ids = np.argmax(logits[0], axis=-1)
output = tokenizer.decode(output_ids, skip_special_tokens=True)
print(output)
'''

print(usage_code)
print("="*80)

