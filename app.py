import json
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import xgrammar as xgr

# ----------------------------
# 模型、XGrammar初始化
# ----------------------------
# 注意：模型名称可以根据你的实际情况替换为合适的模型（建议使用较小模型测试，正式场景可换大模型）
model_name = "meta-llama/Llama-3.1-8B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
model.config.pad_token_id = tokenizer.eos_token_id  # 设置 pad_token_id 避免警告

# 初始化 XGrammar 的基本组件
tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=model.config.vocab_size)
grammar_compiler = xgr.GrammarCompiler(tokenizer_info)

# 默认 JSON schema（以 Person 结构为示例）
default_schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"}
    },
    "required": ["name", "age"]
}
default_schema_text = json.dumps(default_schema, indent=2)


# ----------------------------
# 主转换函数
# ----------------------------
def convert_xml_to_json(xml_input: str, schema_input: str) -> str:
    # 如果用户未提供 JSON schema，则使用默认 schema
    if not schema_input.strip():
        schema_str = default_schema_text
    else:
        schema_str = schema_input.strip()

    # 尝试加载 JSON schema
    try:
        schema = json.loads(schema_str)
    except Exception as e:
        return f"JSON schema 解析错误：{str(e)}"

    # 编译 JSON schema 为 XGrammar Grammar
    try:
        compiled_grammar = grammar_compiler.compile_json_schema(schema)
    except Exception as e:
        return f"编译 JSON schema 出错：{str(e)}"

    # 构造 XGrammar 的 logits processor
    logits_processor = xgr.contrib.hf.LogitsProcessor(compiled_grammar)

    # 构造转换提示，要求 LLM 解析 XML 并输出符合 schema 的 JSON
    prompt = (
        "You are a JSON converter that converts XML data to a structured JSON object.\n"
        "The output must strictly conform to the following JSON schema (and nothing else):\n\n"
        f"{schema_str}\n\n"
        "Convert the following XML to JSON:\n"
        f"{xml_input}\n\n"
        "Output:"
    )

    # 编码 prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    # 调用 generate，并传入 XGrammar logits processor，使生成过程中非法 token 被屏蔽
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=256,
        logits_processor=[logits_processor],
        pad_token_id=tokenizer.eos_token_id,
    )
    # 提取生成部分
    output_text = tokenizer.decode(
        generated_ids[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )
    return output_text.strip()


# ----------------------------
# 构建 Gradio 界面
# ----------------------------
title = "📄 XML to JSON Converter with XGrammar Structure Check"
description = (
    "将任意 XML 转换为 JSON。\n\n"
    "在左侧粘贴 XML 文本，并可选地提供 JSON schema（如果留空，则使用默认结构，示例 schema 为 Person 模式）；\n"
    "系统将调用 LLM 将 XML 转为 JSON，同时利用 XGrammar 限制输出结构，确保生成的 JSON 严格符合 schema。"
)

with gr.Blocks(title=title) as demo:
    gr.Markdown(f"## {title}\n\n{description}")
    
    with gr.Row():
        with gr.Column():
            xml_input = gr.Textbox(lines=12, label="XML 输入", placeholder="在此粘贴 XML 内容…")
            schema_input = gr.Textbox(lines=8, label="JSON Schema（可选）", value=default_schema_text,
                                        placeholder="可提供自定义 JSON schema，否则使用默认 schema")
            convert_btn = gr.Button("转换 XML → JSON")
        with gr.Column():
            json_output = gr.Textbox(lines=12, label="生成的结构化 JSON")
    
    convert_btn.click(
        fn=convert_xml_to_json,
        inputs=[xml_input, schema_input],
        outputs=json_output
    )

if __name__ == "__main__":
    demo.launch()
