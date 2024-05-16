from typing import List, Mapping, Any, Optional
from torchtune.datasets._chat import ChatDataset, ChatFormat
from torchtune.data._types import Message
from torchtune.modules.tokenizers import Tokenizer


def message_converter(sample: Mapping[str, Any], train_on_input: bool = False) -> List[Message]:
    input_msg = sample["instruction"]
    output_msg = sample["output"]

    user_message = Message(
        role="user",
        content=input_msg,
        masked=train_on_input,  # Mask if not training on prompt
    )
    assistant_message = Message(
        role="assistant",
        content=output_msg,
        masked=False,
    )
    # A single turn conversation
    messages = [user_message, assistant_message]

    return messages


def gpt4_self_instruct_german_dataset(
    *,
    tokenizer: Tokenizer,
    max_seq_len: int = 2048,
    chat_format: Optional[ChatFormat] = None,
    train_on_input: bool = False
) -> ChatDataset:

    return ChatDataset(
        tokenizer=tokenizer,
        source="CausalLM/GPT-4-Self-Instruct-German",
        split="train",
        convert_to_messages=message_converter,
        chat_format=chat_format,
        max_seq_len=max_seq_len,
        train_on_input=train_on_input
    )
