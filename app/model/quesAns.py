from transformers import pipeline
from transformers import AutoTokenizer,MambaForCausalLM
# import torch

model_name = "state-spaces/mamba-130m-hf"
tokeniser = AutoTokenizer.from_pretrained(model_name)
model = MambaForCausalLM.from_pretrained(model_name)


def nshot_question(text):
    context = f"""Q: What is capital of USA?\nA: Washington DC\n\nQ: Which is fastest animal?\nA: Cheetah\n\nQ: {text}
    """
    input_ids = tokeniser(context, return_tensors="pt")["input_ids"]
    out = model.generate(input_ids, max_new_tokens=10)
    ans = tokeniser.batch_decode(out)
    temp = ans
    ans = ans[0].split('\n')[-3]
    ans = ans.split("A: ")
    return ans[-1] #, temp

if __name__ == '__main__':
    print( nshot_question("Who is president of America?") )

    print( nshot_question("Who is president of India?") )

    print( nshot_question("Who is president of Russia?") )