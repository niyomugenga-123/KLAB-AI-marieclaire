# Chapter 1: Image Classification with Deep Learning

## Assignment Completed
Trained a deep learning model to classify images into different categories.

## Dataset
- **Categories:** bird, forest, mountain, river
- **Total training images:** ~96 (80% split)
- **Total validation images:** ~24 (20% split)
- **Image size:** 224x224 pixels (resized via item_tfms)

## Model Architecture
- **Model:** ResNet18 (pre-trained on ImageNet)
- **Training method:** Transfer learning (fine-tuning)
- **Epochs:** 4 (plus 1 frozen warm-up epoch)
- **Batch size:** 32

## Results
- Frozen warm-up epoch accuracy: 41.7%
- Final fine-tuned accuracy: 83.3%
- Accuracy plateaued in the last two epochs, likely due to the small
  dataset size (only ~30 images per category)

## Key Learnings
1. **Image size consistency:** all images must be resized to the same
   dimensions (item_tfms=Resize(224)) before batching, or DataLoaders
   throws a shape-mismatch error
2. **Transfer learning:** using a pretrained ResNet18 meant the model only
   needed to learn the last few layers for this specific task, rather
   than learning to recognize images from scratch
3. **Two-stage fine_tune():** fine_tune() first trains only the new final
   layer with the rest frozen (1 epoch), then unfreezes and trains the
   whole network for the remaining epochs
4. **Metric vs. loss:** accuracy is the human-readable metric printed
   each epoch; loss is the actual number the model optimizes internally
5. **Absolute paths matter:** learn.export() saves relative to the
   model's internal path, not the notebook's location, so an absolute
   path avoided a broken relative path error

## How to Use the Saved Model
```python
from fastai.vision.all import *
learn = load_learner('models/image_classifier_model.pkl')
pred_class, pred_idx, probs = learn.predict(PILImage.create('new_image.jpg'))
print(f"Predicted: {pred_class}")
```

---
**Completed:** Chapter 1 - Deep Learning Fundamentals
