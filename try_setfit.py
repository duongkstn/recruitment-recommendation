from datasets import load_dataset
from sentence_transformers.losses import CosineSimilarityLoss

from setfit import SetFitModel, SetFitTrainer

dataset = load_dataset("SetFit/SentEval-CR")
# Select N examples per class (8 in this case)
train_ds = dataset["train"].shuffle(seed=10).select(range(2))
test_ds = dataset["test"]

print(len(train_ds))
for x in train_ds:
    print(x)
print(len(test_ds))


# Load SetFit model from Hub
model = SetFitModel.from_pretrained('intfloat/e5-large', cache_dir="cache_dir")
# model = SetFitModel.from_pretrained('intfloat/multilingual-e5-large', cache_dir="cache_dir")
# model = SetFitModel.from_pretrained('sentence-transformers/paraphrase-mpnet-base-v2', cache_dir="cache_dir")

# Create trainer
trainer = SetFitTrainer(
    model=model,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    loss_class=CosineSimilarityLoss,
    batch_size=16,
    num_iterations=20, # Number of text pairs to generate for contrastive learning
    num_epochs=1, # Number of epochs to use for contrastive learning

)

# Train and evaluate!
trainer.train(end_to_end=False)
metrics = trainer.evaluate()
print(metrics)

print(model.predict(["i am happy", "i am sad"]))

