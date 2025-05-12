# MAIN FILE FOR RELOAD UNLEARNING ON LLMS

# STEP 1: Load the model

# STEP 2: Load the tokenizer

# STEP 3: Define the forget dataset of prompts

# STEP 4: Add in CFT to the forget dataset of prompts

# STEP 5: Add attention collecting hooks to the model

# STEP 6: Generate the forget dataset of responses and save attention activations

# STEP 7: Define the retain dataset of prompts

# STEP 8: Add in CFT to the retain dataset of prompts

# STEP 9: Generate the retain dataset of responses and save attention activations

# STEP 10: Remove attention collecting hooks from the model

# STEP 11: Calculate Knowledge-Values for attention parameters

# STEP 12: Reset the top-k attention parameters

# STEP 13: Train CFT on the retain dataset