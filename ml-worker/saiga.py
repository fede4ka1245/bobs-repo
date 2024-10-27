import torch

from transformers import AutoTokenizer,AutoModelForCausalLM, GenerationConfig
from peft import PeftModel, PeftConfig


LLM_MODEL_NAME = "IlyaGusev/saiga_7b_lora"
DEFAULT_MESSAGE_TEMPLATE = "<s>{role}\n{content}</s>"
DEFAULT_RESPONSE_TEMPLATE = "<s>bot\n"
DEFAULT_SYSTEM_PROMPT = "Ты — Сайга, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им."


# def get_prompt(context: list[str], question: str) -> str:
#     return (f"{context}. На основе полученных данных ответь на запрос '{question}'."
#             f"Ничего не выдумывай, внимательно относись к каждому предложению. Не фантазируй. "
#             f"Если ты не знаешь ответ, скажи, что данных нет и не пытайся что-лтбо выдумать")


def get_prompt(context: list[str], question: str) -> str:
    return f"Контекст: {context}, Вопрос: {question}, Ответ:"


class Conversation:
    def __init__(
            self,
            message_template=DEFAULT_MESSAGE_TEMPLATE,
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            response_template=DEFAULT_RESPONSE_TEMPLATE
    ):
        self.message_template = message_template
        self.response_template = response_template
        self.messages = [{
            "role": "system",
            "content": system_prompt
        }]

    def add_user_message(self, message):
        self.messages.append({
            "role": "user",
            "content": message
        })

    def add_bot_message(self, message):
        self.messages.append({
            "role": "bot",
            "content": message
        })

    def get_prompt(self, tokenizer):
        final_text = ""
        for message in self.messages:
            message_text = self.message_template.format(**message)
            final_text += message_text
        final_text += DEFAULT_RESPONSE_TEMPLATE
        return final_text.strip()


def generate(model, tokenizer, prompt, generation_config):
    data = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    data = {k: v.to(model.device) for k, v in data.items()}
    output_ids = model.generate(
        **data,
        generation_config=generation_config
    )[0]
    output_ids = output_ids[len(data["input_ids"][0]):]
    output = tokenizer.decode(output_ids, skip_special_tokens=True)
    return output.strip()


torch.cuda.empty_cache()
config = PeftConfig.from_pretrained(LLM_MODEL_NAME)
llm_model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    torch_dtype=torch.float16,
    device_map="auto"
)
llm_model = PeftModel.from_pretrained(
    llm_model,
    LLM_MODEL_NAME,
    torch_dtype=torch.float16
)
llm_model.eval()

llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME, use_fast=False)
generation_config = GenerationConfig.from_pretrained(LLM_MODEL_NAME)
generation_config.temperature = 0.2
generation_config.frequency_penalty = 1.7


def get_answer(question: str, context: list[str]):
    prompt = get_prompt(context, question)

    return generate(llm_model, llm_tokenizer, prompt, generation_config)
