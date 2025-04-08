import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import xgrammar as xgr
from pydantic import BaseModel

# ----------------------------
# 1. JSON Schema定义
# ----------------------------
class Person(BaseModel):
    name: str
    age: int

# ----------------------------
# 2. 初始化模型和语法
# ----------------------------
model_name = "meta-llama/Llama-3.1-8B-Instruct"  # 或者换成更小的模型，如 TinyLlama
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
config = model.config

tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=config.vocab_size)
grammar_compiler = xgr.GrammarCompiler(tokenizer_info)
compiled_grammar = grammar_compiler.compile_json_schema(Person)

xgr_logits_processor = xgr.contrib.hf.LogitsProcessor(compiled_grammar)

# ----------------------------
# 3. 主函数：从XML生成JSON
# ----------------------------
def convert_xml_to_json(xml_input: str) -> str:
    prompt = f"""You are a converter that transforms XML into JSON.

Convert the following XML into valid JSON format:

{xml_input}

Output:
"""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=128,
        logits_processor=[xgr_logits_processor]
    )
    generated_text = tokenizer.decode(
        generated_ids[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )
    return generated_text

# ----------------------------
# 4. Gradio界面
# ----------------------------
demo = gr.Interface(
    fn=convert_xml_to_json,
    inputs=gr.Textbox(lines=10, label="Paste your XML here"),
    outputs=gr.Textbox(label="Structured JSON Output"),
    title="🧠 XML ➜ JSON with XGrammar + LLM",
    description="Paste XML content on the left, click 'Run', and get a structurally correct JSON output powered by XGrammar."
)

# ----------------------------
# 5. 运行 Gradio
# ----------------------------
if __name__ == "__main__":
    demo.launch()
