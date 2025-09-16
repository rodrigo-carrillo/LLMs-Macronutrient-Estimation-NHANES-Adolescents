# IMPORT LIBRARIES.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from datasets import load_dataset
import seaborn as sn
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
# from IPython.display import display, Markdown

# Download all shards of the large LLM.
repo_id = "MaziyarPanahi/Mistral-Large-Instruct-2411-GGUF"
model_basename = "Mistral-Large-Instruct-2411.Q5_K_S.gguf"
num_shards = 7
local_dir = "/scratch/rmcarri/Macronutrients_LLMs/LLM_Parts"

#for i in range(1, num_shards + 1):
#   filename = f"{model_basename}-{i:05d}-of-{num_shards:05d}.gguf"
#   hf_hub_download(
#       repo_id = repo_id,
#       filename = filename,
#       local_dir = local_dir,
#       local_dir_use_symlinks = False
#   )
first_shard_path = os.path.join(local_dir, model_basename + "-00001-of-00007.gguf")   # Point to the first part.

# CHECK WHERE THE MODEL WAS STORED (PATH).
print("CHECK WHERE THE MODEL WAS STORED (PATH)")
print(f"Model path: {first_shard_path}")

# LOAD THE LLM (REMEMBER TO UPDATE n_ctx FOR EACH LLM).
print("LOAD THE LLM")
llm = Llama(
    model_path   = first_shard_path,
    n_threads    = 32,        # CPU cores (4)
    n_batch      = 512,       # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
    n_gpu_layers = 40,        # Change this value based on your model and your GPU VRAM pool (43). -1 = all.
    n_ctx        = 131072,    # Context window for LLAMA2=4096; LLAMA3 8B=8192; LLAMA3.3 70B=131072; MISTRAL 7B=32768; DEEPSEEK=4096
    use_mmap     = True,
    use_mlock    = True
)

# READ THE DATASET.
print("READ THE DATASET")
df = pd.read_csv('/scratch/rmcarri/Macronutrients_LLMs/Combined_df_ten_shot_day2_chunk9.csv')
# df = df.sample(n=1000, random_state=42)
# df = df.reset_index(drop = True)
print(df.shape)
print(df.head(5))

# WRITE THE SYSTEM MESSAGE.
print("WRITE THE SYSTEM MESSAGE")
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

# WRITE THE USER MESSAGE.
print("WRITE THE USER MESSAGE") 
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
print("PRINT ONE PROMPT, TO CHECK")
for index, row in tqdm(df.iterrows(), total = df.shape[0], desc = 'Processing Patients'):
    if index == 0:
        prompt = system_message + user_message.format(diet = row['diet'])
        print(prompt)
    else:
        break

# FUNCTION TO GENERATE THE RESPONSE FROM THE LLM.
print("FUNCTION TO GENERATE THE RESPONSES FROM THE LLM")
def lcpp_llm(prompt,
             max_tokens = 1000,
             temperature = 0,
             stop = ["USER"]):
    return llm(prompt, max_tokens = max_tokens, temperature = temperature, stop = stop, echo = False)

# ITERATE THROUGH ALL ROWS TO GENERATE THE RESPONSES FOR EACH OBSERVATION IN THE DATASET.
print("ITERARE TO GET THE RESPONSES FROM THE LLM ON EACH ROW OF THE DATASET")
# Ensure the column exists
df["LLM_answer"] = ""
# Loop and update row-by-row
for index, row in tqdm(df.iterrows(), total = df.shape[0], desc = 'Processing Patients'):
    prompt = system_message + user_message.format(diet = row['diet'])
    response = lcpp_llm(prompt)
    text_response = response['choices'][0]['text'].strip()
    df.at[index, "LLM_answer"] = text_response

# SAVE THE DATASET WITH THE RESPONSES FROM THE LLM.
print("SAVE THE DATASET WITH THE RESPONSES - OUTPUT")
df['model_basename'] = model_basename
model_name = model_basename.split('.gguf')[0]
df.to_csv(f'/scratch/rmcarri/Macronutrients_LLMs/Combined_df_ten_shot_day2_chunk9_{model_name}.csv', index = False)
print("END")