import fiftyone as fo

# train dataset
dataset = fo.zoo.load_zoo_dataset(
              "open-images-v7",
              split="train",
              label_types=["detections"],
              classes=["Plastic bag"],
              max_samples=1000,
          )

# validation dataset
dataset = fo.zoo.load_zoo_dataset(
              "open-images-v7",
              split="validation",
              label_types=["detections"],
              classes=["Plastic bag"],
              max_samples=10,
          )