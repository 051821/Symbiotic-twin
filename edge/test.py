from data.partition import get_edge_partition
from edge.trainer import EdgeTrainer

device_id = "b8:27:eb:bf:9d:51"  # replace with one real device

partition = get_edge_partition(device_id)

trainer = EdgeTrainer(device_id)
trainer.load_data(
    partition["X_train"],
    partition["y_train"],
    partition["X_test"],
    partition["y_test"]
)

trainer.train()
trainer.evaluate()
import pandas as pd

df = pd.read_csv("data/processed_dataset.csv")
print(df["label"].value_counts(normalize=True))
