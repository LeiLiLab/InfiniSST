import torch
import transformers
import conversation as conversation_lib

model_path = ''

tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_path,
            padding_side="right",
            use_fast=False,
        )
tokenizer.pad_token = tokenizer.eos_token
conv = conversation_lib.default_conversation.copy()
roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

conversations = []

conv.messages = []
conv.append_message('USER', '<speech_here> Describe the speech concisely.')
conv.append_message('ASSISTANT', 'sdfg df.')
conversations.append(conv.get_prompt())
print(conversations)

input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="longest",
        truncation=True,
    ).input_ids
targets = input_ids.clone()[0]
print(targets)
print(tokenizer.convert_ids_to_tokens([    1,   887,   526,   365,  5661, 20449, 29892,   263,  2919,  4086,
          322, 18551, 20255, 16370,   491,   501, 29956,  4104,  2285,   399,
        29909,  5667, 12016, 29889,  3492,   526,  2221,   304,  2274,   278,
         7604,  2793,   393,   278,  1404,  8128, 29892,   322,  6985,   278,
         1404,   411,   263, 12875,   310,  9595,   773,  5613,  4086, 29889,
        29943,  2952,   278, 11994, 16112,   322,  5649,   596,  6089,   297,
         9493, 29889,  3148,  1001, 29901,   529,  5965,  5309, 29918,  4150,
        29958, 20355,   915,   278, 12032,  3022,   275,   873, 29889,   319,
         1799,  9047, 13566, 29901,]))
print(tokenizer.convert_ids_to_tokens([269,  2176, 29887,  4489, 29889,     2,]))
sep = conv.sep + conv.roles[1] + ": "
total_len = int(targets.ne(tokenizer.pad_token_id).sum())
rounds = conversations[0].split(conv.sep2)

cur_len = 1
targets[:cur_len] = -1
for i, rou in enumerate(rounds):
    if rou == "":
        break
    parts = rou.split(sep)
    if len(parts) != 2:
        break
    parts[0] += sep
    round_len = len(tokenizer(rou).input_ids)

    instruction_len = len(tokenizer(parts[0]).input_ids) - 2
    print(tokenizer(parts[0]).input_ids)
    print(instruction_len)
    targets[cur_len : cur_len + instruction_len] = -1
    cur_len += round_len

print(targets)

