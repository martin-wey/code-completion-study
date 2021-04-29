## IFT 6010 - Modern Natural Language Processing - Course Project
### Towards unbiased evaluation of DL-based code completion recommender systems
---

This repository contains all the materials used in our experiments. 
The code is meant to be simple and easy-to-use. Each ```.py``` file is a script that can be ran in a CLI.

### Data acquisition

Our dataset is based on the following one: [CodeXGLUE code completion](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/CodeCompletion-token).
All the data can be found in the ```data``` folder. It is structured as follows:

- ```train_java_cs.txt``` - Java/C# functions that can be used to train/fine-tune a model.
- ```train_java_cs_flows.txt``` - Contains code identifiers flows of all the functions of the ```train_java_cs.txt file.
- ```valid_java_cs.txt``` and ```valid_java_cs_flows.txt``` - Validation sets for both token-based and flow-based code representations.


- ```./java/test.java-cs.txt.java``` - Java test set.
- ```./java/test.java-cs.txt.java.flow``` - Java test set using identifiers flows.
- ```./cs/test.java-cs.txt.cs``` - C# test set.
- ```./cs/test.java-cs.txt.cs.flow``` - C# test set using identifiers flows.

*Note that the ```preprocess.py``` and ```code_to_data_flows.py``` scripts can be used to regenerate the complete dataset. We do not explain their utilization as we already provide the data.*

### Model training and fine-tuning

Both model training (*from scratch*) and fine-tuning can be done using the ```run_clm.py``` script.
The notebooks in ```./notebooks/train.ipynb``` and ```./notebooks/fine_tuning.ipynb``` shows how to run both processes in a Google Colab environment. 
The model checkpoint used in both RQ2 and RQ3 can be downloaded [here](https://drive.google.com/drive/folders/1lHlZuO5eCfhfNFo_HevTpiK1B_ZkDxZ2?usp=sharing).

For example, to fine-tune the [CodeGPT model](https://huggingface.co/microsoft/CodeGPT-small-java), the following command can be ran:
```sh
python run_clm.py \
    --model_name_or_path microsoft/CodeGPT-small-java \
    --run_name codegpt-tuned-java-cs \
    --output_dir ./model/codegpt-tuned-java-cs \
    --train_file ./data/train_java_cs.txt \
    --validation_file ./data/valid_java_cs.txt \
    --block_size 1024 \
    --num_train_epochs 20 \
    --gradient_accumulation_steps 4 \
    --learning_rate=3e-5 \
    --weight_decay=0.01 \
    --evaluation_strategy epoch \
    --logging_strategy epoch \
    --save_strategy=epoch \
    --do_train \
    --do_eval \
    --per_device_train_batch_size 4 \
    --report_to wandb
```
The ```--run_name``` and ```--report_to wandb``` arguments allow to report automatically the training and evaluation losses in WanDB. Other hyperparameters of the model can be changed. For the full list of available arguments, you can run ```python run_clm.py --help```. 

### Evaluation

To evaluate a model for the code completion task, one can either used a model on the [HuggingFace Hub](https://huggingface.co/models) or a model saved locally. 
To run the evaluation, you can use the ```eval_lm_completion.py``` script. For example, the following command evaluate a CodeGPT model from the HuggingFace Hub and outputs an evaluation report:

```sh
python eval_lm_completion.py \
    --model_name_or_path microsoft/CodeGPT-small-java \
    --eval_path ./data/java/test.java-cs.txt.java \
    --code_flows_path /data/java/test.java-cs.txt.java.flow \
    --all_tokens False
```
The ```--model_name_or_path``` can be changed to point to a local model by providing its path. 
When ```--all_tokens``` is set to False, the script will evaluate the model on different type of tokens and provide detailed results. If it is set to True, then it will only provide the global accuracy of the model on all types of tokens.

*Note that even though the evaluation is made on the token-based dataset, the identifiers flows are required to distinguish each token's type and provide a detailed evaluation report.*
