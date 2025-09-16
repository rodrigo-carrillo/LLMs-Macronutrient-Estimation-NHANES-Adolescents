# Load original dataset and split into train, validation and test.

print("Load datasets.")
import pandas as pd

df_train = pd.read_csv('/scratch/rmcarri/Macronutrients_LLMs_FineTuning/Combined_df_ten_shot_day2_chunk1.csv')
df_val = pd.read_csv('/scratch/rmcarri/Macronutrients_LLMs_FineTuning/Combined_df_ten_shot_day2_chunk2.csv')

# df_train = df_train.sample(n = 5, random_state = 42)
# df_val = df_val.sample(n = 5, random_state = 42)

df_train['Expected_Output'] = df_train[['DRxIKCAL', 'DRxIPROT', 'DRxICARB', 'DRxISUGR', 'DRxIFIBE', 'DRxITFAT']].astype(str).agg('; '.join, axis=1)
df_val['Expected_Output'] = df_val[['DRxIKCAL', 'DRxIPROT', 'DRxICARB', 'DRxISUGR', 'DRxIFIBE', 'DRxITFAT']].astype(str).agg('; '.join, axis=1)

print(df_train.shape)
print(df_val.shape)


# Import libraries.

print("Import libraries.")
from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
import torch
import evaluate
from tqdm import tqdm
import pandas as pd
from datasets import Dataset
from tqdm import tqdm
from transformers import TrainingArguments, EarlyStoppingCallback, DataCollatorForSeq2Seq
from trl import SFTTrainer


# Load the vanilla model.

print("Load vanilla model from unsloth.")
model, tokenizer = FastLanguageModel.from_pretrained(
    # from_tf=True,
    model_name="unsloth/Mistral-Small-24B-Instruct-2501",
    max_seq_length=32768,
    dtype = None,
    load_in_4bit = True,
    load_in_8bit = False,
)


# Original system message.

print("Original system message.")
system_message = """
SYSTEM:

You are a highly experienced clinical dietitian and nutrition scientist with advanced training in macronutrient metabolism, dietary pattern analysis, and expert proficiency with the Nutrition Data System for Research (NDSR).
Your task is to analyze a patient's 24-hour dietary recall and estimate the following six nutritional values:
1. Total energy (kcal)
2. Total protein (g)
3. Total carbohydrates (g)
4. Total sugars (g)
5. Total dietary fiber (g)
6. Total fat (g)


You must base your estimates **solely** on the foods listed in the dietary recall. Each food is formatted as:
- 'FOOD NAME (grams or milliliters)'
- Foods are separated by semicolons.


You must reason internally using the following process:
---
Chain-of-Thought Reasoning:
1. **Identify Foods and Portions**  
Parse the list of foods and quantities (in grams/mL), identifying each food type and its contribution to energy and macronutrients.

2. **Reference Internal Knowledge from NDSR**  
Use your mental simulation of the NDSR nutrient database to retrieve nutrient values per 100g for each food item, adjusting proportionally based on the reported intake.

3. **Estimate Nutrient Totals**  
Accumulate values for each nutrient: total kcal, protein, carbohydrate, sugars, fiber, and fat.

4. **Format Your Output**  
Round your final estimates to two decimal places and report them in the exact required order and format.
---


Critical Output Rules:
- Your output must be exactly six numeric values separated by semicolons. No additional text, comments, labels, or formatting is allowed.
- If your answer contains anything else, immediately reprint the six values only.
- Any output that deviates from this exact six-number, semicolon-separated format will be considered invalid.
- Failure to follow this format will be considered an invalid output.
- Do **not** include units, labels, or introductory phrases.
- Do **not** infer unlisted foods or make assumptions beyond what is provided.
- You must **always** return a complete prediction, even with minimal input.
- Output **only** the six values, separated by semicolons, in this exact order: 'kcal; protein; carbohydrate; sugars; fiber; fat'
- If any nutrient is likely to be zero based on the input, explicitly return '0' for that value.
- Do **not** include missing values, comments, or formatting variations.


Examples for Calibration:
Patient Input:
24-hour dietary recall: MILK, LOW FAT (1%) (76.25); BEEF, NS AS TO CUT, COOKED, NS AS TO FAT EATEN (12.56); BEEF, NS AS TO CUT, COOKED, LEAN ONLY EATEN (134); BEEF, NS AS TO CUT, COOKED, LEAN ONLY EATEN (134); TORTILLA, CORN (168); CEREAL, READY-TO-EAT, NFS (52.5); APPLE JUICE, 100% (325.5); POTATO, NFS (120); BROCCOLI, COOKED, FROM FRESH, FAT NOT ADDED IN COOKING (117); BROCCOLI, COOKED, FROM FRESH, FAT NOT ADDED IN COOKING (117); CARROTS, COOKED, FROM FRESH, FAT NOT ADDED IN COOKING (117); SOFT DRINK, FRUIT FLAVORED, CAFFEINE FREE (248)
Expected Output: 1630; 107.97; 233.28; 79.83; 27.7; 33.68

Patient Input:
24-hour dietary recall: ICE CREAM, REGULAR, NOT CHOCOLATE (141.31); CHEESE, NFS (24); BOLOGNA, NFS (28); SUNFLOWER SEEDS, HULLED, ROASTED, SALTED (46); BREAD, WHITE (52); COOKIE, MARSHMALLOW, W/ RICE CEREAL (NO-BAKE) (60); MILK 'N CEREAL BAR (24); PASTA W/ TOMATO SAUCE & MEAT/MEATBALLS, CANNED (280.13); SOFT DRINK, FRUIT-FLAVORED, CAFFEINE FREE (368)
Expected Output: 1629; 43.29; 205.67; 113.29; 14.9; 74.29

Patient Input:
24-hour dietary recall: CHICKEN NUGGETS, FROM FROZEN (96); CHICKEN TENDERS OR STRIPS, BREADED, FROM SCHOOL LUNCH (80); BIG MAC (MCDONALDS) (135); MACARONI OR NOODLES WITH CHEESE, MADE FROM PACKAGED MIX (57.5); APPLE, RAW (125); STRAWBERRIES, RAW (108); POTATO, FRENCH FRIES, FAST FOOD (55); POTATO, MASHED, FROM SCHOOL LUNCH (62.5); WATER, BOTTLED, PLAIN (20); WATER, BOTTLED, PLAIN (345)
Expected Output: 1293; 48.28; 135.41; 29.22; 13.2; 62.15

Patient Input:
24-hour dietary recall: MILK, COW'S, FLUID, 2% FAT (259.25); CHICKEN, THIGH, STEWED, W/ SKIN (88); BREAD, GARLIC (333); RICE, WHITE, COOKED, REGULAR, NO FAT ADD IN COOKING (79); FROSTED FLAKES, KELLOGG'S (74.31); PIZZA, CHEESE, THIN CRUST (136.78); PLUM, RAW (66); GRAPE JUICE (332.06); FRUIT JUICE DRINK (449.5); FRUIT JUICE DRINK (449.5)
Expected Output: 2923; 81.63; 443.26; 206.48; 14.8; 93.63

Patient Input:
24-hour dietary recall: ICE CREAM CONE, VANILLA, PREPACKAGED (95); CHICKEN, NS AS TO PART AND COOKING METHOD, SKIN NOT EATEN (75.94); RICE, WHITE, COOKED, NO ADDED FAT (138.25); TACO, MEAT, NO CHEESE (180); CARROTS, RAW (45); TOMATOES, RAW (67.5); LETTUCE, RAW (19.69); SOFT DRINK, COLA (264.5); SOFT DRINK, COLA (264.5); WATER, BOTTLED, PLAIN (1740)
Expected Output: 1338; 57.38; 162.67; 81.11; 8; 51.38

Patient Input:
24-hour dietary recall: GENERAL TSO CHICKEN (866.88); WAFFLE, FRUIT (78); RICE, FRIED, W/ PORK (210.38); SYRUP, DIETETIC (5); GRAPE JUICE DRINK (250)
Expected Output: 2473; 129.47; 215.26; 71.96; 9.1; 121.12

Patient Input:
24-hour dietary recall: MILK, COW'S, FLUID, 1% FAT (533.75); MILK, SOY, READY-TO-DRINK, NOT BABY (535.94); CHEESE, NATURAL, CHEDDAR OR AMERICAN TYPE (56.7); HAM, SLICED, PREPACKAGED OR DELI, LUNCHEON MEAT (56); CHEESEBURGER, W/ MAYO & TOMATO/CATSUP, ON BUN CHEESEBURGER, (314); EGGS, WHOLE, FRIED (INCL SCRAMBLED, NO MILK ADDED) (46); PEANUT BUTTER (32); PEANUT BUTTER (32); BREAD, RYE (50); BREAD, RYE (25); COOKIE, OATMEAL, W/ RAISINS OR DATES (39); OATMEAL, CKD, INST, MADE W/ MILK, FAT NOT ADDED IN COOKING (307.13); RICE, WHITE, COOKED, REGULAR, NO FAT ADD IN COOKING (207.38); RICE W/ BEANS AND BEEF (433.19); WHITE POTATO, BOILED, W/O PEEL, NS AS TO FAT (516); TOMATOES, RAW (40); LETTUCE, RAW (24); SNICKERS CANDY BAR (17); WATER, TAP (9480)
Expected Output: 4270; 201.78; 503.17; 127.52; 38.9; 164.7

Patient Input:
24-hour dietary recall: ICE CREAM, REGULAR, NOT CHOCOLATE (141.31); FISH STICK/FILLET, NS TYPE, FLOURED/BREADED, FRIED (51); WHITE POTATO, FRENCH FRIES, FROM FROZEN, DEEP-FRIED (60.56); TOMATO CATSUP (15); TOMATO CATSUP (15); FRUIT JUICE DRINK, W/ VIT B1 & VIT C (546.88); WATER, BOTTLED, UNSWEETENED (518.44); WATER, BOTTLED, UNSWEETENED (518.44)
Expected Output: 854; 18.29; 126.89; 73.54; 4.5; 31.65

Patient Input:
24-hour dietary recall: MILK, LOW FAT (1%) (106.75); PORK, CRACKLINGS, COOKED (51.19); PINTO/CALICO/RED MEX BEANS, DRY, CKD, FAT ADD, NS TYPE FAT (100.13); TORTILLA, FLOUR (WHEAT) (225); FRUITY PEBBLES CEREAL (52.5); APPLE, RAW (182); WHITE POTATO, CHIPS, RESTRUCTURED, BAKED (21); SOFT DRINK, FRUIT-FLAVORED, W/ CAFFEINE (241.5); WATER, BOTTLED, UNSWEETENED (2610)
OutExpected Outputput: 1693; 51.67; 257.79; 83.32; 22; 51.17

Patient Input:
24-hour dietary recall: PUDDING, TAPIOCA, MADE FROM DRY MIX, MADE WITH MILK (299.06); OYSTERS, COOKED, NS AS TO COOKING METHOD (81.81); BEEF WITH VEGETABLES EXCLUDING CARROTS, BROCCOLI, AND DARK-G (132.28); PORK AND VEGETABLES EXCLUDING  CARROTS, BROCCOLI, AND DARK-G (132.28); RICE, WHITE, COOKED, NS AS TO FAT ADDED IN COOKING (213.94); BEEF NOODLE SOUP, CANNED OR READY-TO-SERVE (808.25); TEA, ICED, INSTANT, BLACK, DECAFFEINATED, PRE-SWEETENED WITH (333.5); SOFT DRINK, COLA, DECAFFEINATED (372); SOFT DRINK, FRUIT FLAVORED, CAFFEINE FREE (372); WATER, BOTTLED, UNSWEETENED (720)
Expected Output: 1742; 58.25; 278.91; 167.56; 7.9; 45.46


Begin reasoning internally and return your prediction in the exact required output format.
Do not explain your reasoning.
Do not repeat or preface the answer.
Output only the final six numbers in this format: kcal; protein; carbohydrate; sugars; fiber; fat.
Do not prefix with "Assistant:" or "Answer:".
Output the six values ONCE and nothing else. Failure to follow this format will be considered incorrect.

"""


# Original user message.

print("Original user message.")
user_message = """

USER:

Please analyze the patient's dietary intake and return the six requested nutrition estimates.
Patient Input:
24-hour dietary recall: {diet}

Return only the six numeric values in this format:
1234.56; 78.90; 123.45; 67.89; 10.00; 50.00
Do not include any text, explanations, or extra formatting. Only output the six numbers, separated by semicolons, rounded to two decimals.
Do not explain your reasoning.
Do not repeat or preface the answer.
Output only the final six numbers in this format: kcal; protein; carbohydrate; sugars; fiber; fat.
Do not prefix with "Assistant:" or "Answer:".
Output the six values ONCE and nothing else. Failure to follow this format will be considered incorrect.

"""


# PRINT THE PROMPT, TO CHECK.

print("PRINT ONE PROMPT, TO CHECK.")
for index, row in tqdm(df_train.iterrows(), total = df_train.shape[0], desc = 'Processing Patients'):
    if index == 0:
        prompt = system_message + user_message.format(diet = row['diet'])
        print(prompt)
    else:
        break


# Predictions with the original vanilla model.

print("Prediction with the vanilla model.")
predicted_summaries = []

for index, row in tqdm(df_val.iterrows(), total = df_val.shape[0], desc = 'Processing Abstracts Vanilla Model'):

    try:
        prompt = system_message + user_message.format(diet = row['diet'])
        inputs = tokenizer(prompt, return_tensors = "pt").to("cuda")

        outputs = model.generate(
            **inputs,
            max_new_tokens = 4096,
            use_cache = True,
            temperature = 1e-5,
            pad_token_id = tokenizer.eos_token_id
        )

        prediction = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[-1]:],
            skip_special_tokens = True,
            cleanup_tokenization_spaces = True
        )

        predicted_summaries.append(prediction)

    except Exception as e:
        print(e) # log error and continue
        continue


# Check the output and store in the test dataframe.

print("Store output in test dataset.")
print(len(predicted_summaries))
df_val['LLM_Original'] = predicted_summaries


# Fine-tuning begins.


# Generate training and validation datasets.

print("Generate datasets for fine tuning.")
training_dict = df_train.astype(str).to_dict(orient='list')
training_dataset = Dataset.from_dict(training_dict)

validation_dict = df_val.astype(str).to_dict(orient='list')
validation_dataset = Dataset.from_dict(validation_dict)

EOS_TOKEN = tokenizer.eos_token


# Alpaca prompt.

print("Alpaca prompt")
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}


### Input:
USER:

Please analyze the patient's dietary intake and return the six requested nutrition estimates.
Patient Input:
24-hour dietary recall: {diet_recall}

Return only the six numeric values in this format:
1234.56; 78.90; 123.45; 67.89; 10.00; 50.00
Do not include any text, explanations, or extra formatting. Only output the six numbers, separated by semicolons, rounded to two decimals.
Do not explain your reasoning.
Do not repeat or preface the answer.
Output only the final six numbers in this format: kcal; protein; carbohydrate; sugars; fiber; fat.
Do not prefix with "Assistant:" or "Answer:".
Output the six values ONCE and nothing else. Failure to follow this format will be considered incorrect.


### Response:
{nutrition}

"""


# Prompt formatter for training and validation datasets.

print("Prompt formatter for training and validation sets.")
def prompt_formatter(example, prompt_template):
    instruction="""
SYSTEM:

You are a highly experienced clinical dietitian and nutrition scientist with advanced training in macronutrient metabolism, dietary pattern analysis, and expert proficiency with the Nutrition Data System for Research (NDSR).
Your task is to analyze a patient's 24-hour dietary recall and estimate the following six nutritional values:
1. Total energy (kcal)
2. Total protein (g)
3. Total carbohydrates (g)
4. Total sugars (g)
5. Total dietary fiber (g)
6. Total fat (g)


You must base your estimates **solely** on the foods listed in the dietary recall. Each food is formatted as:
- 'FOOD NAME (grams or milliliters)'
- Foods are separated by semicolons.


You must reason internally using the following process:
---
Chain-of-Thought Reasoning:
1. **Identify Foods and Portions**  
Parse the list of foods and quantities (in grams/mL), identifying each food type and its contribution to energy and macronutrients.

2. **Reference Internal Knowledge from NDSR**  
Use your mental simulation of the NDSR nutrient database to retrieve nutrient values per 100g for each food item, adjusting proportionally based on the reported intake.

3. **Estimate Nutrient Totals**  
Accumulate values for each nutrient: total kcal, protein, carbohydrate, sugars, fiber, and fat.

4. **Format Your Output**  
Round your final estimates to two decimal places and report them in the exact required order and format.
---


Critical Output Rules:
- Your output must be exactly six numeric values separated by semicolons. No additional text, comments, labels, or formatting is allowed.
- If your answer contains anything else, immediately reprint the six values only.
- Any output that deviates from this exact six-number, semicolon-separated format will be considered invalid.
- Failure to follow this format will be considered an invalid output.
- Do **not** include units, labels, or introductory phrases.
- Do **not** infer unlisted foods or make assumptions beyond what is provided.
- You must **always** return a complete prediction, even with minimal input.
- Output **only** the six values, separated by semicolons, in this exact order: 'kcal; protein; carbohydrate; sugars; fiber; fat'
- If any nutrient is likely to be zero based on the input, explicitly return '0' for that value.
- Do **not** include missing values, comments, or formatting variations.


Examples for Calibration:
Patient Input:
24-hour dietary recall: MILK, LOW FAT (1%) (76.25); BEEF, NS AS TO CUT, COOKED, NS AS TO FAT EATEN (12.56); BEEF, NS AS TO CUT, COOKED, LEAN ONLY EATEN (134); BEEF, NS AS TO CUT, COOKED, LEAN ONLY EATEN (134); TORTILLA, CORN (168); CEREAL, READY-TO-EAT, NFS (52.5); APPLE JUICE, 100% (325.5); POTATO, NFS (120); BROCCOLI, COOKED, FROM FRESH, FAT NOT ADDED IN COOKING (117); BROCCOLI, COOKED, FROM FRESH, FAT NOT ADDED IN COOKING (117); CARROTS, COOKED, FROM FRESH, FAT NOT ADDED IN COOKING (117); SOFT DRINK, FRUIT FLAVORED, CAFFEINE FREE (248)
Expected Output: 1630; 107.97; 233.28; 79.83; 27.7; 33.68

Patient Input:
24-hour dietary recall: ICE CREAM, REGULAR, NOT CHOCOLATE (141.31); CHEESE, NFS (24); BOLOGNA, NFS (28); SUNFLOWER SEEDS, HULLED, ROASTED, SALTED (46); BREAD, WHITE (52); COOKIE, MARSHMALLOW, W/ RICE CEREAL (NO-BAKE) (60); MILK 'N CEREAL BAR (24); PASTA W/ TOMATO SAUCE & MEAT/MEATBALLS, CANNED (280.13); SOFT DRINK, FRUIT-FLAVORED, CAFFEINE FREE (368)
Expected Output: 1629; 43.29; 205.67; 113.29; 14.9; 74.29

Patient Input:
24-hour dietary recall: CHICKEN NUGGETS, FROM FROZEN (96); CHICKEN TENDERS OR STRIPS, BREADED, FROM SCHOOL LUNCH (80); BIG MAC (MCDONALDS) (135); MACARONI OR NOODLES WITH CHEESE, MADE FROM PACKAGED MIX (57.5); APPLE, RAW (125); STRAWBERRIES, RAW (108); POTATO, FRENCH FRIES, FAST FOOD (55); POTATO, MASHED, FROM SCHOOL LUNCH (62.5); WATER, BOTTLED, PLAIN (20); WATER, BOTTLED, PLAIN (345)
Expected Output: 1293; 48.28; 135.41; 29.22; 13.2; 62.15

Patient Input:
24-hour dietary recall: MILK, COW'S, FLUID, 2% FAT (259.25); CHICKEN, THIGH, STEWED, W/ SKIN (88); BREAD, GARLIC (333); RICE, WHITE, COOKED, REGULAR, NO FAT ADD IN COOKING (79); FROSTED FLAKES, KELLOGG'S (74.31); PIZZA, CHEESE, THIN CRUST (136.78); PLUM, RAW (66); GRAPE JUICE (332.06); FRUIT JUICE DRINK (449.5); FRUIT JUICE DRINK (449.5)
Expected Output: 2923; 81.63; 443.26; 206.48; 14.8; 93.63

Patient Input:
24-hour dietary recall: ICE CREAM CONE, VANILLA, PREPACKAGED (95); CHICKEN, NS AS TO PART AND COOKING METHOD, SKIN NOT EATEN (75.94); RICE, WHITE, COOKED, NO ADDED FAT (138.25); TACO, MEAT, NO CHEESE (180); CARROTS, RAW (45); TOMATOES, RAW (67.5); LETTUCE, RAW (19.69); SOFT DRINK, COLA (264.5); SOFT DRINK, COLA (264.5); WATER, BOTTLED, PLAIN (1740)
Expected Output: 1338; 57.38; 162.67; 81.11; 8; 51.38

Patient Input:
24-hour dietary recall: GENERAL TSO CHICKEN (866.88); WAFFLE, FRUIT (78); RICE, FRIED, W/ PORK (210.38); SYRUP, DIETETIC (5); GRAPE JUICE DRINK (250)
Expected Output: 2473; 129.47; 215.26; 71.96; 9.1; 121.12

Patient Input:
24-hour dietary recall: MILK, COW'S, FLUID, 1% FAT (533.75); MILK, SOY, READY-TO-DRINK, NOT BABY (535.94); CHEESE, NATURAL, CHEDDAR OR AMERICAN TYPE (56.7); HAM, SLICED, PREPACKAGED OR DELI, LUNCHEON MEAT (56); CHEESEBURGER, W/ MAYO & TOMATO/CATSUP, ON BUN CHEESEBURGER, (314); EGGS, WHOLE, FRIED (INCL SCRAMBLED, NO MILK ADDED) (46); PEANUT BUTTER (32); PEANUT BUTTER (32); BREAD, RYE (50); BREAD, RYE (25); COOKIE, OATMEAL, W/ RAISINS OR DATES (39); OATMEAL, CKD, INST, MADE W/ MILK, FAT NOT ADDED IN COOKING (307.13); RICE, WHITE, COOKED, REGULAR, NO FAT ADD IN COOKING (207.38); RICE W/ BEANS AND BEEF (433.19); WHITE POTATO, BOILED, W/O PEEL, NS AS TO FAT (516); TOMATOES, RAW (40); LETTUCE, RAW (24); SNICKERS CANDY BAR (17); WATER, TAP (9480)
Expected Output: 4270; 201.78; 503.17; 127.52; 38.9; 164.7

Patient Input:
24-hour dietary recall: ICE CREAM, REGULAR, NOT CHOCOLATE (141.31); FISH STICK/FILLET, NS TYPE, FLOURED/BREADED, FRIED (51); WHITE POTATO, FRENCH FRIES, FROM FROZEN, DEEP-FRIED (60.56); TOMATO CATSUP (15); TOMATO CATSUP (15); FRUIT JUICE DRINK, W/ VIT B1 & VIT C (546.88); WATER, BOTTLED, UNSWEETENED (518.44); WATER, BOTTLED, UNSWEETENED (518.44)
Expected Output: 854; 18.29; 126.89; 73.54; 4.5; 31.65

Patient Input:
24-hour dietary recall: MILK, LOW FAT (1%) (106.75); PORK, CRACKLINGS, COOKED (51.19); PINTO/CALICO/RED MEX BEANS, DRY, CKD, FAT ADD, NS TYPE FAT (100.13); TORTILLA, FLOUR (WHEAT) (225); FRUITY PEBBLES CEREAL (52.5); APPLE, RAW (182); WHITE POTATO, CHIPS, RESTRUCTURED, BAKED (21); SOFT DRINK, FRUIT-FLAVORED, W/ CAFFEINE (241.5); WATER, BOTTLED, UNSWEETENED (2610)
OutExpected Outputput: 1693; 51.67; 257.79; 83.32; 22; 51.17

Patient Input:
24-hour dietary recall: PUDDING, TAPIOCA, MADE FROM DRY MIX, MADE WITH MILK (299.06); OYSTERS, COOKED, NS AS TO COOKING METHOD (81.81); BEEF WITH VEGETABLES EXCLUDING CARROTS, BROCCOLI, AND DARK-G (132.28); PORK AND VEGETABLES EXCLUDING  CARROTS, BROCCOLI, AND DARK-G (132.28); RICE, WHITE, COOKED, NS AS TO FAT ADDED IN COOKING (213.94); BEEF NOODLE SOUP, CANNED OR READY-TO-SERVE (808.25); TEA, ICED, INSTANT, BLACK, DECAFFEINATED, PRE-SWEETENED WITH (333.5); SOFT DRINK, COLA, DECAFFEINATED (372); SOFT DRINK, FRUIT FLAVORED, CAFFEINE FREE (372); WATER, BOTTLED, UNSWEETENED (720)
Expected Output: 1742; 58.25; 278.91; 167.56; 7.9; 45.46


Begin reasoning internally and return your prediction in the exact required output format.
Do not explain your reasoning.
Do not repeat or preface the answer.
Output only the final six numbers in this format: kcal; protein; carbohydrate; sugars; fiber; fat.
Do not prefix with "Assistant:" or "Answer:".
Output the six values ONCE and nothing else. Failure to follow this format will be considered incorrect.

"""
    diet_recall=example["diet"]
    nutrition=example["Expected_Output"]

    formatted_prompt = prompt_template.format(instruction=instruction,
                                              diet_recall=diet_recall,
                                              nutrition=nutrition) + EOS_TOKEN

    return {'formatted_prompt': formatted_prompt}


formatted_training_dataset = training_dataset.map(
    prompt_formatter,
    fn_kwargs={'prompt_template': alpaca_prompt}
)


print(formatted_training_dataset['formatted_prompt'][0])


formatted_validation_dataset = validation_dataset.map(
    prompt_formatter,
    fn_kwargs={'prompt_template': alpaca_prompt}
)


print(formatted_validation_dataset['formatted_prompt'][0])


# Get PEFT model.

print("Get PEFT model.")
model_fine_tuned = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing = True,
    random_state = 42,
    loftq_config = None
)


model_fine_tuned


# Implement the trainer.

print("Trainer.")
trainer = SFTTrainer(
    model = model_fine_tuned,
    tokenizer = tokenizer,
    train_dataset = formatted_training_dataset,
    eval_dataset = formatted_validation_dataset,
    dataset_text_field = "formatted_prompt",
    max_seq_length = 32768,
    data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
    dataset_num_proc = 1,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        num_train_epochs = 10, # Set this for 1 full training run.
        # max_steps = 10,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # Use this for WandB etc
    ),
)


# Training

print("Training.")
training_history = trainer.train()


# Save the fine-tuned model.

print("Save and name the fine-tuned model")
lora_model_name = "Mistral_Small_24B_Instruct_2501_finetuned"
model_fine_tuned.save_pretrained(lora_model_name)


# Load the fine-tuned model.

print("Load the fine-tuned model")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = lora_model_name,
    max_seq_length = 32768,
    dtype = None,
    load_in_4bit = True,
    load_in_8bit = False
)


model


# Make predictions with the fine-tuned model.

print("Test the fine-tuned model making predictions in chunk 2.")
predicted_summaries_ft = []
for index, row in tqdm(df_val.iterrows(), total = df_val.shape[0], desc = 'Processing Abstracts Fine-Tuned Model'):

    try:
        prompt = system_message + user_message.format(diet = row['diet'])
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        outputs = model.generate(
            **inputs,
            max_new_tokens = 4096,
            use_cache = True,
            temperature = 1e-5,
            pad_token_id = tokenizer.eos_token_id
        )

        prediction = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[-1]:],
            skip_special_tokens = True,
            cleanup_tokenization_spaces = True
        )

        predicted_summaries_ft.append(prediction)

    except Exception as e:
        print(e) # log error and continue
        continue


# Check the outputs of the fine-tuned model and store in the test dataframe.

print("Store output in test dataset.")
print(len(predicted_summaries_ft))
print(predicted_summaries_ft)
df_val['LLM_Fine_Tuned'] = predicted_summaries_ft


# Save dataframe
df_val.to_csv('/scratch/rmcarri/Macronutrients_LLMs_FineTuning/df_val_with_answers_chunk2.csv')





####################################################################################################################
print("Test the fine-tuned model making predictions in chunk 3.")

del df_val, validation_dict, validation_dataset, formatted_validation_dataset
df_val = pd.read_csv('/scratch/rmcarri/Macronutrients_LLMs_FineTuning/Combined_df_ten_shot_day2_chunk3.csv')
# df_val = df_val.sample(n = 5, random_state = 42)
df_val['Expected_Output'] = df_val[['DRxIKCAL', 'DRxIPROT', 'DRxICARB', 'DRxISUGR', 'DRxIFIBE', 'DRxITFAT']].astype(str).agg('; '.join, axis=1)
validation_dict = df_val.astype(str).to_dict(orient='list')
validation_dataset = Dataset.from_dict(validation_dict)
formatted_validation_dataset = validation_dataset.map(
    prompt_formatter,
    fn_kwargs={'prompt_template': alpaca_prompt}
)



predicted_summaries_ft = []
for index, row in tqdm(df_val.iterrows(), total = df_val.shape[0], desc = 'Processing Abstracts Fine-Tuned Model'):

    try:
        prompt = system_message + user_message.format(diet = row['diet'])
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        outputs = model.generate(
            **inputs,
            max_new_tokens = 4096,
            use_cache = True,
            temperature = 1e-5,
            pad_token_id = tokenizer.eos_token_id
        )

        prediction = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[-1]:],
            skip_special_tokens = True,
            cleanup_tokenization_spaces = True
        )

        predicted_summaries_ft.append(prediction)

    except Exception as e:
        print(e) # log error and continue
        continue


# Check the outputs of the fine-tuned model and store in the test dataframe.

print("Store output in test dataset.")
print(len(predicted_summaries_ft))
print(predicted_summaries_ft)
df_val['LLM_Fine_Tuned'] = predicted_summaries_ft


# Save dataframe
df_val.to_csv('/scratch/rmcarri/Macronutrients_LLMs_FineTuning/df_val_with_answers_chunk3.csv')
####################################################################################################################





####################################################################################################################
print("Test the fine-tuned model making predictions in chunk 4.")

del df_val, validation_dict, validation_dataset, formatted_validation_dataset
df_val = pd.read_csv('/scratch/rmcarri/Macronutrients_LLMs_FineTuning/Combined_df_ten_shot_day2_chunk4.csv')
# df_val = df_val.sample(n = 5, random_state = 42)
df_val['Expected_Output'] = df_val[['DRxIKCAL', 'DRxIPROT', 'DRxICARB', 'DRxISUGR', 'DRxIFIBE', 'DRxITFAT']].astype(str).agg('; '.join, axis=1)
validation_dict = df_val.astype(str).to_dict(orient='list')
validation_dataset = Dataset.from_dict(validation_dict)
formatted_validation_dataset = validation_dataset.map(
    prompt_formatter,
    fn_kwargs={'prompt_template': alpaca_prompt}
)



predicted_summaries_ft = []
for index, row in tqdm(df_val.iterrows(), total = df_val.shape[0], desc = 'Processing Abstracts Fine-Tuned Model'):

    try:
        prompt = system_message + user_message.format(diet = row['diet'])
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        outputs = model.generate(
            **inputs,
            max_new_tokens = 4096,
            use_cache = True,
            temperature = 1e-5,
            pad_token_id = tokenizer.eos_token_id
        )

        prediction = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[-1]:],
            skip_special_tokens = True,
            cleanup_tokenization_spaces = True
        )

        predicted_summaries_ft.append(prediction)

    except Exception as e:
        print(e) # log error and continue
        continue


# Check the outputs of the fine-tuned model and store in the test dataframe.

print("Store output in test dataset.")
print(len(predicted_summaries_ft))
print(predicted_summaries_ft)
df_val['LLM_Fine_Tuned'] = predicted_summaries_ft


# Save dataframe
df_val.to_csv('/scratch/rmcarri/Macronutrients_LLMs_FineTuning/df_val_with_answers_chunk4.csv')
####################################################################################################################





####################################################################################################################
print("Test the fine-tuned model making predictions in chunk 5.")

del df_val, validation_dict, validation_dataset, formatted_validation_dataset
df_val = pd.read_csv('/scratch/rmcarri/Macronutrients_LLMs_FineTuning/Combined_df_ten_shot_day2_chunk5.csv')
# df_val = df_val.sample(n = 5, random_state = 42)
df_val['Expected_Output'] = df_val[['DRxIKCAL', 'DRxIPROT', 'DRxICARB', 'DRxISUGR', 'DRxIFIBE', 'DRxITFAT']].astype(str).agg('; '.join, axis=1)
validation_dict = df_val.astype(str).to_dict(orient='list')
validation_dataset = Dataset.from_dict(validation_dict)
formatted_validation_dataset = validation_dataset.map(
    prompt_formatter,
    fn_kwargs={'prompt_template': alpaca_prompt}
)



predicted_summaries_ft = []
for index, row in tqdm(df_val.iterrows(), total = df_val.shape[0], desc = 'Processing Abstracts Fine-Tuned Model'):

    try:
        prompt = system_message + user_message.format(diet = row['diet'])
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        outputs = model.generate(
            **inputs,
            max_new_tokens = 4096,
            use_cache = True,
            temperature = 1e-5,
            pad_token_id = tokenizer.eos_token_id
        )

        prediction = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[-1]:],
            skip_special_tokens = True,
            cleanup_tokenization_spaces = True
        )

        predicted_summaries_ft.append(prediction)

    except Exception as e:
        print(e) # log error and continue
        continue


# Check the outputs of the fine-tuned model and store in the test dataframe.

print("Store output in test dataset.")
print(len(predicted_summaries_ft))
print(predicted_summaries_ft)
df_val['LLM_Fine_Tuned'] = predicted_summaries_ft


# Save dataframe
df_val.to_csv('/scratch/rmcarri/Macronutrients_LLMs_FineTuning/df_val_with_answers_chunk5.csv')
####################################################################################################################





####################################################################################################################
print("Test the fine-tuned model making predictions in chunk 6.")

del df_val, validation_dict, validation_dataset, formatted_validation_dataset
df_val = pd.read_csv('/scratch/rmcarri/Macronutrients_LLMs_FineTuning/Combined_df_ten_shot_day2_chunk6.csv')
# df_val = df_val.sample(n = 5, random_state = 42)
df_val['Expected_Output'] = df_val[['DRxIKCAL', 'DRxIPROT', 'DRxICARB', 'DRxISUGR', 'DRxIFIBE', 'DRxITFAT']].astype(str).agg('; '.join, axis=1)
validation_dict = df_val.astype(str).to_dict(orient='list')
validation_dataset = Dataset.from_dict(validation_dict)
formatted_validation_dataset = validation_dataset.map(
    prompt_formatter,
    fn_kwargs={'prompt_template': alpaca_prompt}
)



predicted_summaries_ft = []
for index, row in tqdm(df_val.iterrows(), total = df_val.shape[0], desc = 'Processing Abstracts Fine-Tuned Model'):

    try:
        prompt = system_message + user_message.format(diet = row['diet'])
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        outputs = model.generate(
            **inputs,
            max_new_tokens = 4096,
            use_cache = True,
            temperature = 1e-5,
            pad_token_id = tokenizer.eos_token_id
        )

        prediction = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[-1]:],
            skip_special_tokens = True,
            cleanup_tokenization_spaces = True
        )

        predicted_summaries_ft.append(prediction)

    except Exception as e:
        print(e) # log error and continue
        continue


# Check the outputs of the fine-tuned model and store in the test dataframe.

print("Store output in test dataset.")
print(len(predicted_summaries_ft))
print(predicted_summaries_ft)
df_val['LLM_Fine_Tuned'] = predicted_summaries_ft


# Save dataframe
df_val.to_csv('/scratch/rmcarri/Macronutrients_LLMs_FineTuning/df_val_with_answers_chunk6.csv')
####################################################################################################################





####################################################################################################################
print("Test the fine-tuned model making predictions in chunk 7.")

del df_val, validation_dict, validation_dataset, formatted_validation_dataset
df_val = pd.read_csv('/scratch/rmcarri/Macronutrients_LLMs_FineTuning/Combined_df_ten_shot_day2_chunk7.csv')
# df_val = df_val.sample(n = 5, random_state = 42)
df_val['Expected_Output'] = df_val[['DRxIKCAL', 'DRxIPROT', 'DRxICARB', 'DRxISUGR', 'DRxIFIBE', 'DRxITFAT']].astype(str).agg('; '.join, axis=1)
validation_dict = df_val.astype(str).to_dict(orient='list')
validation_dataset = Dataset.from_dict(validation_dict)
formatted_validation_dataset = validation_dataset.map(
    prompt_formatter,
    fn_kwargs={'prompt_template': alpaca_prompt}
)



predicted_summaries_ft = []
for index, row in tqdm(df_val.iterrows(), total = df_val.shape[0], desc = 'Processing Abstracts Fine-Tuned Model'):

    try:
        prompt = system_message + user_message.format(diet = row['diet'])
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        outputs = model.generate(
            **inputs,
            max_new_tokens = 4096,
            use_cache = True,
            temperature = 1e-5,
            pad_token_id = tokenizer.eos_token_id
        )

        prediction = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[-1]:],
            skip_special_tokens = True,
            cleanup_tokenization_spaces = True
        )

        predicted_summaries_ft.append(prediction)

    except Exception as e:
        print(e) # log error and continue
        continue


# Check the outputs of the fine-tuned model and store in the test dataframe.

print("Store output in test dataset.")
print(len(predicted_summaries_ft))
print(predicted_summaries_ft)
df_val['LLM_Fine_Tuned'] = predicted_summaries_ft


# Save dataframe
df_val.to_csv('/scratch/rmcarri/Macronutrients_LLMs_FineTuning/df_val_with_answers_chunk7.csv')
####################################################################################################################





####################################################################################################################
print("Test the fine-tuned model making predictions in chunk 8.")

del df_val, validation_dict, validation_dataset, formatted_validation_dataset
df_val = pd.read_csv('/scratch/rmcarri/Macronutrients_LLMs_FineTuning/Combined_df_ten_shot_day2_chunk8.csv')
# df_val = df_val.sample(n = 5, random_state = 42)
df_val['Expected_Output'] = df_val[['DRxIKCAL', 'DRxIPROT', 'DRxICARB', 'DRxISUGR', 'DRxIFIBE', 'DRxITFAT']].astype(str).agg('; '.join, axis=1)
validation_dict = df_val.astype(str).to_dict(orient='list')
validation_dataset = Dataset.from_dict(validation_dict)
formatted_validation_dataset = validation_dataset.map(
    prompt_formatter,
    fn_kwargs={'prompt_template': alpaca_prompt}
)



predicted_summaries_ft = []
for index, row in tqdm(df_val.iterrows(), total = df_val.shape[0], desc = 'Processing Abstracts Fine-Tuned Model'):

    try:
        prompt = system_message + user_message.format(diet = row['diet'])
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        outputs = model.generate(
            **inputs,
            max_new_tokens = 4096,
            use_cache = True,
            temperature = 1e-5,
            pad_token_id = tokenizer.eos_token_id
        )

        prediction = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[-1]:],
            skip_special_tokens = True,
            cleanup_tokenization_spaces = True
        )

        predicted_summaries_ft.append(prediction)

    except Exception as e:
        print(e) # log error and continue
        continue


# Check the outputs of the fine-tuned model and store in the test dataframe.

print("Store output in test dataset.")
print(len(predicted_summaries_ft))
print(predicted_summaries_ft)
df_val['LLM_Fine_Tuned'] = predicted_summaries_ft


# Save dataframe
df_val.to_csv('/scratch/rmcarri/Macronutrients_LLMs_FineTuning/df_val_with_answers_chunk8.csv')
####################################################################################################################





####################################################################################################################
print("Test the fine-tuned model making predictions in chunk 9.")

del df_val, validation_dict, validation_dataset, formatted_validation_dataset
df_val = pd.read_csv('/scratch/rmcarri/Macronutrients_LLMs_FineTuning/Combined_df_ten_shot_day2_chunk9.csv')
# df_val = df_val.sample(n = 5, random_state = 42)
df_val['Expected_Output'] = df_val[['DRxIKCAL', 'DRxIPROT', 'DRxICARB', 'DRxISUGR', 'DRxIFIBE', 'DRxITFAT']].astype(str).agg('; '.join, axis=1)
validation_dict = df_val.astype(str).to_dict(orient='list')
validation_dataset = Dataset.from_dict(validation_dict)
formatted_validation_dataset = validation_dataset.map(
    prompt_formatter,
    fn_kwargs={'prompt_template': alpaca_prompt}
)



predicted_summaries_ft = []
for index, row in tqdm(df_val.iterrows(), total = df_val.shape[0], desc = 'Processing Abstracts Fine-Tuned Model'):

    try:
        prompt = system_message + user_message.format(diet = row['diet'])
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        outputs = model.generate(
            **inputs,
            max_new_tokens = 4096,
            use_cache = True,
            temperature = 1e-5,
            pad_token_id = tokenizer.eos_token_id
        )

        prediction = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[-1]:],
            skip_special_tokens = True,
            cleanup_tokenization_spaces = True
        )

        predicted_summaries_ft.append(prediction)

    except Exception as e:
        print(e) # log error and continue
        continue


# Check the outputs of the fine-tuned model and store in the test dataframe.

print("Store output in test dataset.")
print(len(predicted_summaries_ft))
print(predicted_summaries_ft)
df_val['LLM_Fine_Tuned'] = predicted_summaries_ft


# Save dataframe
df_val.to_csv('/scratch/rmcarri/Macronutrients_LLMs_FineTuning/df_val_with_answers_chunk9.csv')
####################################################################################################################





####################################################################################################################
print("Test the fine-tuned model making predictions in chunk 10.")

del df_val, validation_dict, validation_dataset, formatted_validation_dataset
df_val = pd.read_csv('/scratch/rmcarri/Macronutrients_LLMs_FineTuning/Combined_df_ten_shot_day2_chunk10.csv')
# df_val = df_val.sample(n = 5, random_state = 42)
df_val['Expected_Output'] = df_val[['DRxIKCAL', 'DRxIPROT', 'DRxICARB', 'DRxISUGR', 'DRxIFIBE', 'DRxITFAT']].astype(str).agg('; '.join, axis=1)
validation_dict = df_val.astype(str).to_dict(orient='list')
validation_dataset = Dataset.from_dict(validation_dict)
formatted_validation_dataset = validation_dataset.map(
    prompt_formatter,
    fn_kwargs={'prompt_template': alpaca_prompt}
)



predicted_summaries_ft = []
for index, row in tqdm(df_val.iterrows(), total = df_val.shape[0], desc = 'Processing Abstracts Fine-Tuned Model'):

    try:
        prompt = system_message + user_message.format(diet = row['diet'])
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        outputs = model.generate(
            **inputs,
            max_new_tokens = 4096,
            use_cache = True,
            temperature = 1e-5,
            pad_token_id = tokenizer.eos_token_id
        )

        prediction = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[-1]:],
            skip_special_tokens = True,
            cleanup_tokenization_spaces = True
        )

        predicted_summaries_ft.append(prediction)

    except Exception as e:
        print(e) # log error and continue
        continue


# Check the outputs of the fine-tuned model and store in the test dataframe.

print("Store output in test dataset.")
print(len(predicted_summaries_ft))
print(predicted_summaries_ft)
df_val['LLM_Fine_Tuned'] = predicted_summaries_ft


# Save dataframe
df_val.to_csv('/scratch/rmcarri/Macronutrients_LLMs_FineTuning/df_val_with_answers_chunk10.csv')
####################################################################################################################





print('THE END!')