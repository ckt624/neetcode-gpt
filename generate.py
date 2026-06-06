import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution:
    def generate(self, model, new_chars: int, context: TensorType[int], context_length: int, int_to_char: dict) -> str:
        # 1. Crop context to context_length if it exceeds it: context[:, -context_length:]
        # 2. Run model(context) -> take last position's logits -> apply softmax(dim=-1)
        # 3. Sample next token with torch.multinomial(probs, 1, generator=generator)
        # 4. Append sampled token to context with torch.cat
        # 5. Map token to character using int_to_char and accumulate result
        # Do not alter the fixed code below — it ensures reproducible test output.

        generator = torch.manual_seed(0)
        initial_state = generator.get_state()
        ans = []

        for i in range(new_chars):
            # Work even if context.shape[1] > context_length:
            context = context[:, -context_length:] # batch size=1, sequence length of int

            y = model(context)
            prob = nn.functional.softmax(y[:, -1, :], dim=-1) # softmax over feature dim
            next_char = torch.multinomial(prob, 1, generator=generator)
            # The line where you call torch.multinomial(). Pass in the generator as well.
            generator.set_state(initial_state)
            context = torch.cat([context, next_char], dim=-1) # concat along last dim
            ans.append(int_to_char[next_char.item()])


        return ''.join(ans)
        # Once your code passes the test, check out the Colab link to see your code generate new Drake lyrics!
