from datasets import load_dataset
dataset = load_dataset('acidtib/tcg-magic-cards')

print(dataset['train'].column_names)
print(dataset['train'][0])
