from transformers import AutoModelWithLMHead, AutoTokenizer
import torch


tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelWithLMHead.from_pretrained("microsoft/DialoGPT-medium")

class SmallTalk:
    """ Keep conversation going """

    def __init__(self, model=model, tokenizer=tokenizer):
        self.tokenizer = tokenizer
        self.model = model
        self.chat_history_ids = None
        self.params_gen = {"max_length": 1000, "pad_token_id": self.tokenizer.eos_token_id}

    def encode_text(self, text):
        text = text + self.tokenizer.eos_token
        out = self.tokenizer.encode(text, return_tensors='pt')
        return out

    def retrieve_last_message(self, bot_input_ids):
        last_message = self.chat_history_ids[:, bot_input_ids.shape[-1]:][0]
        return self.tokenizer.decode(last_message, skip_special_tokens=True)

    def talk(self, text):
        # encode the new user input, add the eos_token and return a tensor in Pytorch
        new_user_input_ids = self.encode_text(text)

        # append the new user input tokens to the chat history
        if self.chat_history_ids != None:
            bot_input_ids = torch.cat([self.chat_history_ids, new_user_input_ids], dim=-1)
        else:
            bot_input_ids = new_user_input_ids

        # generated a response while limiting the total chat history to 1000 tokens,
        self.chat_history_ids = self.model.generate(bot_input_ids, **self.params_gen)
        return self.retrieve_last_message(bot_input_ids)

class DumbAgent:
    """ replies to user with his own phrase """

    def __init__(self):
        pass

    def talk(self, x):
        return f"Everyone says {x}. Buy an elephant!"
