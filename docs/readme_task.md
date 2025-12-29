
# Codex Instructions: Generate README.md for This Computer Vision Project

You are an AI coding assistant working inside this repository.  
Your job is to **read the existing codebase (including subfolders)** and then **write a complete `README.md`** for a computer vision course project.

The README is for TAs and classmates who will clone this repo, understand the method, and reproduce the experiments.

---

## 1. Files and structure you must inspect

Recursively scan the current folder and pay attention to:

- Training and evaluation scripts  
  - Examples: `train.py`, `train_*.py`, `eval_val.py`, `eval_test.py`, `eval_aigcdetect.py`, `eval_all_val.sh`, `eval_all_test.sh`
- Model definitions  
  - Examples: `models.py`, `aigcnet*.py`, `TwoBranchAIGCNet`, `AIGCNetSmall`, `AIGCNetLarge`
- Dataset and transforms  
  - Examples: `dataset.py`, `data/`, `build_transforms`, `AIGCDataset`, dataset root config and CLI flags
- Utility modules  
  - Examples: `utils.py`, `dataset_paths.py`, logging helpers, seed setting
- Saved weights and result files  
  - `weights/` subfolders, `best_model.pth`, CSV files with metrics such as `*_val.csv`, `*_test.csv`
- Any configuration files or shell scripts  
  - Examples: `.env.example`, `run_*.sh`, `eval_all_*.sh`, `requirements.txt`, `environment.yml`

Use actual function names, CLI arguments, and directory paths from the code. Do not invent scripts or options that do not exist.

---

## 2. Target audience and project context

In the README, assume the reader is a **computer vision student** or **course TA** who:

- Knows basic deep learning and PyTorch
- Has access to a Linux machine with at least one GPU, but the project is designed for **small GPUs** and can also run on CPU for small tests
- Needs to understand how this project explores **lightweight CNNs for AI generated image detection**

The course rules for this project:

- **No pretrained models** are allowed (no ImageNet, no external checkpoints)
- **No external training data** beyond the provided course dataset
- The focus is on:
  - Designing and analyzing **ultra lightweight CNN backbones**
  - Exploring input channels such as **RGB**, **gradient (Sobel)**, and optionally **FFT based frequency channels**
  - Studying **regularization and data augmentation** (strong aug, label smoothing, mixup)
  - Evaluating **distribution shift** on an external benchmark of generative models

Make sure these constraints are clearly described in the README introduction.

---

## 3. README high level outline

Generate a `README.md` with the following sections and order:

1. **Project Title**

   A concise title that reflects the project, similar to:

   > Lightweight RGB plus Gradient CNNs for AI Generated Image Detection

   Use the actual project name if there is one already in the repo.

2. **Overview**

   A short paragraph describing:

   - The task: supervised binary classification of **real vs AI generated images**
   - The course constraints  
     - no pretrained models  
     - no extra training data  
   - The main idea:
     - small residual CNN backbones (AIGCNetSmall and AIGCNetLarge)
     - single branch vs two branch design
     - RGB plus gradient channels, and optional FFT channels
   - Mention that the project evaluates both:
     - a validation split created from the training set  
     - an out of distribution test setting (for example AIGCDetectBenchmark)

3. **Key Features**

   A bullet list with items such as:

   - Ultra lightweight CNNs with about 2.76M and 5.51M parameters
   - Multi channel inputs: RGB, Sobel gradient, and optionally FFT magnitude
   - Single branch and two branch architectures
   - Ablations on regularization: strong augmentation, label smoothing, mixup
   - Evaluation under dataset shift on multiple generator families

   Use numbers and model names derived from the code.

4. **Repository Structure**

   A tree style overview similar to:

   ```text
   .
   ├── train.py
   ├── eval_val.py
   ├── eval_test.py
   ├── eval_aigcdetect.py
   ├── dataset.py
   ├── models.py
   ├── utils.py
   ├── weights/
   │   └── <experiment_name>/
   │       ├── best_model.pth
   │       └── *.csv
   └── scripts/
       └── eval_all_*.sh
    ```

Match the actual file layout and names in this repo.

5. **Environment and Requirements**

   * Describe Python and PyTorch versions based on `requirements.txt`, `environment.yml`, comments in code, or typical defaults (for example Python 3.10, PyTorch 2.x with CUDA).
   * List major dependencies: `torch`, `torchvision`, `numpy`, `opencv-python` or `Pillow`, `tqdm`, `scikit-learn` if used.
   * Provide sample commands:

     ```bash
     conda create -n cvpj python=3.10
     conda activate cvpj
     pip install -r requirements.txt
     ```

     If there is no requirements file, infer a minimal `pip install` command from imports.

6. **Dataset Preparation**

   Describe:

   * The expected directory layout under `data` or the path passed via `--data_dir`.

   * How the course dataset is organized, for example:

     ```text
     data/
       train/
         0_real/
         1_fake/
       test/
         0_real/
         1_fake/
     ```

     or use the exact structure defined in `AIGCDataset`.

   * If there is a script that converts labels into `0_real` and `1_fake` folders, explain how to run it.

   * Mention any environment variables or `.env` files used for dataset paths.

7. **Model Architecture**

   Provide a textual summary of the main models using information from `models.py`:

   * **Single branch backbone (AIGCNetSmall / AIGCNetLarge)**

     * Stem conv block
     * Three residual stages with downsampling (for example 32 to 64 to 128 to 256)
     * Global average pooling and a linear classifier to 2 logits
     * Channel configurations:

       * `rgb` (3 channels)
       * `rgb_fft` (4 channels)
       * `rgb_grad` (5 channels)
       * `rgb_fft_grad` (6 channels)
   * **Two branch backbone (TwoBranchAIGCNet)**

     * Separate RGB and gradient backbones with shared structure
     * Features concatenated then passed through a final linear classifier
     * Typical parameter counts (approx 5.51M for the small two branch model)

   Use diagrams in text, bullet lists, and simple dimension notation, not full mathematical formulae.

8. **Training**

   Show how to train the main models:

   * Provide command line examples, using actual arguments and defaults from `train.py`, for example:

     ```bash
     python train.py \
       --data_dir /path/to/dataset \
       --model_size small \
       --channel_mode rgb_grad \
       --two_branch \
       --epochs 50 \
       --batch_size 64 \
       --lr 3e-4 \
       --run_name extra_twobranch_small_mixup_only
     ```

   * Explain the core training options:

     * `--channel_mode` controls which channels are used
     * `--two_branch` switches from single branch to two branch architecture
     * Flags that enable strong augmentation, label smoothing, mixup etc

   * Mention that the training script will:

     * create a subfolder in `weights/` for each run
     * save `best_model.pth`
     * log metrics to CSV or text files if implemented

9. **Validation Evaluation**

   Describe how to re run evaluation on the validation split reconstructed from the train set:

   * Show a simple example using `eval_val.py`:

     ```bash
     python eval_val.py \
       --data_dir /path/to/dataset \
       --checkpoint weights/extra_twobranch_small_mixup_only/best_model.pth \
       --channel_mode rgb_grad \
       --model_size small \
       --two_branch
     ```

   * Explain that:

     * the script reconstructs the train and validation split using `--val_ratio` and `--seed`
     * it reports accuracy, precision, recall, F1, TP, TN, FP, FN
     * metrics are also saved to a CSV file next to the checkpoint

10. **Test and OOD Evaluation**

    Provide instructions for:

    * Evaluating on the course test set:

      ```bash
      python eval_test.py \
        --data_dir /path/to/dataset \
        --checkpoint weights/<run_name>/best_model.pth \
        --channel_mode rgb_grad \
        --model_size small \
        --two_branch
      ```

      Use the actual script name and flags.

    * Evaluating on the external generator benchmark (for example `eval_aigcdetect.py`):

      * Explain the expected folder structure of the benchmark dataset.
      * Show how to run evaluation to get per generator accuracies and an overall mean.
      * Mention any options for handling nested folders.

    * If there are helper shell scripts such as `eval_all_val.sh` or `eval_all_test.sh`, include a short explanation and example usage.

11. **Results**

    Summarize the key findings using actual numbers from CSVs in `weights/` where possible. For example, include:

    * A **validation leaderboard** table that compares:

      * different channel modes
      * single branch vs two branch
      * presence or absence of strong augmentation, label smoothing, mixup
      * accuracy, precision, recall, F1
    * A **test / OOD table** that shows:

      * performance of the validation best model vs regularized models with strong augmentation
      * highlight the observation that:

        * some models with the best in domain accuracy do not generalize well to the OOD test set
        * models trained with stronger augmentation can have lower validation accuracy but higher OOD accuracy

    Use markdown tables and boldface to highlight the best numbers in each column.

12. **Reproducing Our Main Experiments**

    Provide a short checklist for reproducing the main results:

    1. Install environment.
    2. Prepare dataset folders.
    3. Run training command for the main model.
    4. Run validation evaluation.
    5. Run test and benchmark evaluation.
    6. Inspect the generated CSVs and logs.

13. **Limitations and Future Work**

    Add a brief bullet list, for example:

    * The models are intentionally small due to course constraints.
    * No external data and no pretrained weights were used.
    * ViT style architectures were not explored because they typically need large scale pretraining.
    * Possible extensions: better frequency representations, more advanced architectures, self supervised pretraining on synthetic data.

14. **Acknowledgments**

    * Mention the course name, instructor, and any reference baselines or public benchmarks which inspired the project.
    * If this repository is adapted from another open source project, clearly credit the original repo.

---

## 4. Writing style and formatting

When you generate `README.md`:

* Use clear section headings with `##` and `###`.
* Use concise paragraphs, bullet lists, and code blocks with shell commands.
* Keep the tone professional but accessible for students.
* Never refer to this instruction file by name. The README should read as a standalone document.
* Do not mention that you are an AI model. Write as if the project authors wrote the README themselves.
* Avoid fabricating scripts, options, or directories. Everything you describe must actually exist in the current repository, except for high level conceptual summaries (which are based on this instruction file).

---

## 5. Final deliverable

After reading the whole repository and this instruction file:

* Create a single file named **`README.md`** in the repository root.
* The content should follow the outline above, filled with project specific details discovered from the codebase.
* Aim for a length of roughly **one to three pages** of markdown, detailed enough for full reproduction but not excessively long.

Once `README.md` is generated, your task is complete.


