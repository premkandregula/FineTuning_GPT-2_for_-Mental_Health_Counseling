# FineTuning_GPT-2_for_Mental_Health_Counseling
This repository contains a Jupyter Notebook (`FineTuneGPT2_with_mental_health_counselling.ipynb`) that demonstrates how to fine-tune the GPT-2 language model on a mental health counseling dataset to generate empathetic and contextually relevant responses to mental health-related questions. The notebook also includes code to integrate the fine-tuned model with a Telegram bot for interactive question-answering.

## Project Overview

This project fine-tunes the GPT-2 model (specifically the `gpt2` variant) using a mental health counseling dataset to create a conversational model capable of answering questions related to mental health. The fine-tuned model is saved and can be used to generate responses to user queries. Additionally, the project includes a Telegram bot implementation to interact with the model in real-time.

The notebook is implemented using Python, leveraging the Hugging Face `transformers` and `datasets` libraries for model fine-tuning and the `pyTelegramBotAPI` library for bot integration.

## Dataset

The dataset used for fine-tuning is sourced from Hugging Face's dataset hub (`0x22almostEvil/reasoning-gsm-qna-oa`). It contains question-answer pairs related to mental health counseling. Additionally, custom CSV files (`question.csv` for training and `valid.csv` for validation) are used to format the data into question-answer pairs for fine-tuning.

## Dependencies

To run the notebook, you need the following Python packages:

- `transformers==4.28.0`
- `datasets`
- `pandas`
- `pyTelegramBotAPI`
- `openai==0.28`

You can install these dependencies using the following commands:

```bash
pip install transformers==4.28.0
pip install datasets
pip install pyTelegramBotAPI
pip install openai==0.28
```

## Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. **Set Up a Virtual Environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   Run the dependency installation commands listed above.

4. **Prepare the Dataset**:
   - Download or prepare the `question.csv` and `valid.csv` files containing question-answer pairs.
   - Place these files in the same directory as the notebook or update the file paths in the code.

5. **Hugging Face Login**:
   - The notebook uses the Hugging Face Hub to access the dataset and model. Log in using your Hugging Face token:
     ```python
     from huggingface_hub import login
     login()
     ```
   - You can obtain a token from [Hugging Face](https://huggingface.co/settings/tokens).

6. **Run the Notebook**:
   Open the notebook in Jupyter or Google Colab and execute the cells sequentially.

## Usage

1. **Fine-Tuning the Model**:
   - The notebook loads the GPT-2 model and tokenizer, processes the CSV files into question-answer pairs, and fine-tunes the model using the Hugging Face `Trainer` API.
   - The fine-tuned model is saved to the `fine_tuned_mentalhealth_counselling_gpt2` directory.

2. **Asking Questions**:
   - Use the `ask_question` function to query the fine-tuned model:
     ```python
     question = "Every winter I find myself getting sad because of the weather. How can I fight this?"
     answer = ask_question(question, fine_tuned_model, tokenizer)
     print(f"Question: {question}\nAnswer: {answer}")
     ```
   - Example questions and responses are included in the notebook.

3. **Telegram Bot**:
   - The notebook includes code to set up a Telegram bot using `pyTelegramBotAPI`. To use it:
     - Obtain a Telegram Bot Token from [BotFather](https://t.me/BotFather).
     - Update the bot token in the code.
     - Run the bot code to interact with the fine-tuned model via Telegram.

## Training Details

- **Model**: GPT-2 (`gpt2` variant)
- **Training Parameters**:
  - Epochs: 3
  - Batch Size: 8 (training and evaluation)
  - Evaluation Steps: 100
  - Save Steps: 100
  - Save Total Limit: 3
- **Optimizer**: AdamW (with a deprecated warning, suggesting the use of PyTorch's `torch.optim.AdamW`)
- **Training Output**:
  - The training process completes 114 steps with a training loss of approximately 2.42 and a validation loss of 2.15 at step 100.

## Model Evaluation

- The model is evaluated during training using a validation dataset (`valid.csv`).
- The validation loss is logged every 100 steps, with a recorded value of 2.146175 at step 100.
- Example responses to mental health questions are provided in the notebook, demonstrating the model's ability to generate contextually relevant answers, though some responses may need further refinement for accuracy and coherence.

## Telegram Bot Integration

The notebook includes a basic Telegram bot setup using `pyTelegramBotAPI`. The bot processes incoming messages and responds using the fine-tuned GPT-2 model. To extend the bot:

- Add more robust error handling.
- Implement additional commands for user interaction.
- Deploy the bot on a server for continuous operation (e.g., using Heroku or a VPS).
