# Smart Waste Sorter

Week 1 of the Smart Waste Sorter computer vision project: classify waste photos into six classes (cardboard, glass, metal, paper, plastic, trash) using the TrashNet dataset.

## What is in this folder
| File | Purpose |
|---|---|
| `00_setup_and_data.ipynb` | Saves the helper file, downloads TrashNet, removes duplicates and builds the fixed train / validation / test split (70 / 15 / 15, seed 42) |
| `01_cnn.ipynb` | Small CNN from scratch, ResNet-18 transfer learning, error analysis, data augmentation stretch and the week 1 findings note |
| `waste_sorter/ws_utils.py` | Shared helpers (data loaders, training loop, metrics, result logging) |
| `waste_sorter/results.csv` | Logged results of each experiment |
| `waste_sorter/data/` | TrashNet images (2,524 after removing 3 duplicates) and `split.csv` |

## Results (TrashNet test set, 379 images)
| Model | Test accuracy | Macro-F1 | Trash F1 | Local photos accuracy |
|---|---|---|---|---|
| Small CNN (from scratch, 128 px, 15 epochs) | 0.686 | 0.633 | 0.308 | 0.267 |
| ResNet-18 fine-tuned (224 px, 5 epochs) | 0.937 | 0.926 | 0.850 | 0.283 |
| ResNet-18 + data augmentation (224 px, 8 epochs) | 0.923 | 0.915 | 0.857 | 0.367 |

## Key findings
- Pretraining on ImageNet gave a large gain over the small CNN trained from scratch (+0.251 accuracy, +0.542 trash F1).
- The most frequent ResNet-18 errors are cardboard → paper and glass → plastic: smooth boxes look like paper, and transparent or frosted glass looks like clear plastic.
- Accuracy on 60 local photos (not included in this repository) drops sharply for every model, which shows a domain shift between TrashNet's clean photos and real-world images. Data augmentation reduced the gap a little.
- The augmentation run used 8 epochs on a CPU, while the plain ResNet-18 used 5 epochs on a GPU, so that comparison is not fully controlled.

## How to run
1. Run `00_setup_and_data.ipynb` once to create the data and split (or use the included `waste_sorter/data`).
2. Run `01_cnn.ipynb` in Google Colab (T4 GPU) or locally. Locally, keep the `waste_sorter` folder next to the notebooks.
3. Requirements: `torch`, `torchvision`, `pandas`, `scikit-learn`, `matplotlib`, `pillow`.
