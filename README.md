# A Contrastive Learning Neural Model for Math Word Problems

This repository is the [PyTorch](http://pytorch.org/) implementation for the paper:

> [Seeking Patterns, Not just Memorizing Procedures: Contrastive Learning for Solving Math Word Problems](https://arxiv.org/abs/2110.08464)

## A Contrastive Learning Model

![image-20211021183615670](https://tva1.sinaimg.cn/large/008i3skNly1gvn464lfjfj60u00ujjuu02.jpg)


## Requirements

- python 3
- [PyTorch](http://pytorch.org/) 1.8
- [transformers](https://huggingface.co/) 4.9.1


## Usage

- **Data download**
  - [Our processed data](https://drive.google.com/file/d/1AZJrSH8AHxysdTUttH_e0xOHGFlAfPFm/view?usp=sharing)
    - Chinese dataset: Math_23K
    - English dataset: MathQA
    - Data-processing codes are also provided in `tools/`

- **Install transformer library**

  ```
  pip install transformers
  ```

- **Pretrained bert download**

| data-pair       | zh-zh                 | en-en                                                        | zh-en                                                        |
| --------------- | --------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| pretrained bert | [bert-base-chinese]() | [bert-base-uncased](https://huggingface.co/bert-base-uncased) | [bert-base-multilingual-uncased](https://huggingface.co/bert-base-multilingual-uncased) |

- **Emplace vocab_list**

  Emplace the original `vocab.txt` in pretrained bert model with our `vocab.txt` in **pretrained_models_vocab/**

- **Directory structure**

  Organize above-mentioned files like this.

  ```
  .
  ├── data/
  │   ├── MathQA_bert_token_test.json
  │   ├── ...
  │   └── pairs/
  │       ├── MathQA-MathQA-sample.json
  │       └── ...
  ├── pretrained_models/
  │   └── bert-base-chinese/
  │       ├── vocab.txt(emplaced)
  │       └── ...
  ├── src/
  ├── tools/
  ├── run_cl.py
  ├── run_ft.py
  ├──...

- **Train**

  We provide some train shell scripts, and give an example. If you want to write your own train script, please see the code for more details.

  - **Train stage 1: contrastive learning**

    | train contrastive learning | mono-lingual-zh              | mono-lingual-en              | multi-lingual              |
    | -------------------------- | ---------------------------- | ---------------------------- | -------------------------- |
    | shell script               | ./train-cl-monolingual-zh.sh | ./train-cl-monolingual-en.sh | ./train-cl-multilingual.sh |

  - **Train stage 2: finetune**

    | finetune     | mono-lingual-zh              | mono-lingual-en              | multi-lingual-zh              | multi-lingual-en              |
    | ------------ | ---------------------------- | ---------------------------- | ----------------------------- | ----------------------------- |
    | shell script | ./train-ft-monolingual-zh.sh | ./train-ft-monolingual-en.sh | ./train_ft_multilingual-zh.sh | ./train_ft_multilingual-en.sh |

  - **An example**

    To train a multi-lingual contrastive learning model. You can first run this shell. The model will be saved to `./output`

    ```
    bash train-cl-multilingual.sh
    ```

    Then you can finetune the above model in one specific language, using

    ```
    bash train_ft_multilingual-zh.sh
    ```

    

## Results

| Model                              | Accuracy (Math 23K) | Accuracy (MathQA) |
| ---------------------------------- | ------------------- | ----------------- |
| Monolingual Setting BERT-TD w CL   | 83.2%               | 76.3%             |
| Multilingual Setting mBERT-TD w CL | 83.9%               | 76.3%             |



## Citation

