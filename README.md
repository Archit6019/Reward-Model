# RLHF Reward Model Trainer

This repository contains code for training a reward model used in Reinforcement Learning from Human Feedback (RLHF). The reward model is based on GPT-2 and is trained on a dataset of human preferences.

## Overview
This project implements a reward model for RLHF using the following components:

1. A GPT-2 based model for generating rewards
2. A custom dataset loader using the Hugging Face datasets library
3. A training loop with a specialized loss function
4. Inference functionality to compare responses

## Installation
To use this code, you'll need to install the required dependencies.
```bash
pip install -r requirements.txt
```

## Usage
1. For training the reward model
```bash
ins = Reward_Model()

model = ins.GPT2()

ins.train(model=model, epochs=20)
```

2. Use the trained model for inference 
```bash
loaded_model = ins.load_model('reward_model')

prompt = "What is Depreciation"
response_a = "Depreciation is the drop in value of an asset due to wear and tear, age and obsolescence (going out of date) as recorded in an organization's financial records."
response_b = """ 
What is Depreciation – 10 Important Facts to Know?
When a business buys a new asset, the purchase price of that asset is depreciated over time to reflect its usage and eventual obsolescence. Depreciation expense can be a tax deductible expense and is usually a non-cash expense reported on a company’s income statement and balance sheet. The amount of depreciation expense a company reports each year is the difference between the original purchase price of the asset and what the current value of that asset might be. Here are 10 important facts to know about depreciation:
1. Depreciation is a non-cash expense. It is an expense that is reported in a business’s income statement and balance sheet and not a cash flow expense.
2. Depreciation is an accounting standard and it is required to be disclosed in a business’s financial statements.
3. The amount of depreciation is usually a tax expense and not a cash expense reported on a company’s income statement
"""
    
ins.inference(prompt, response_a, response_b, loaded_model)
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

