# bachelor-thesis

## Things to change if you want to add a new model
1. Init Training:
   - Add the model to model.py
   - Add the model to train_versions folder
   - Add the model to the model_architectures folder

2. Init Evaluation:
   - Add the model to evaluate.py
   - Add the model to the eval_configs.json

3. Training Bevahior:
  - Set use_lr_finder to True in the train.py in order to find the best learning rate for fine-tuning
  - Set the found learning rate in the train.py


X. Add classes:
 - Search for 'num_classes=' and replace the number
 - Search for 'num_classes": ' and replace the number