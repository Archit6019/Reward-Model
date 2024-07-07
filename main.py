from reward import Reward_Model

# ------For training 
ins = Reward_Model()
model = ins.GPT2()
ins.train(model=model, epochs=20)

# ------For inference 
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