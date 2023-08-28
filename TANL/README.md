# TANL

## Summary of findings
1. During the MUC trigger annotation process, we often would see several good candidate
triggers for a template. Hence, it is probably best to not penalize the model if it
selects one of many good triggers. We get around this by transforming single documents
into multiple training examples using combinations of triggers (over the templates).
This modification resulted in small performance improvements over all metrics when 
using 2 or 3 triggers per template, but then performance regressed with 4 triggers.
2. Originally, entities were tagged by their role filler type, triggers by the event
type, and relationships were of the form "role filler type: event type". However,
we found that making all the entities be tagged with the same label (ex: "template entity"),
triggers be tagged with a more verbose description (ex: "trigger for attack event"), and
relationships be tagged with a more verbose description (ex: "target argument for attack event")
led to better performance. This might be due to the model making better use of tags
that more closesly resemble natural language. This resulted in small improvements
over all metrics.
3. After implementing a callback to have the model predict on several training and
evaluation split examples, then computing the template level errors, we found that
the model was capable of improving on the training split, but not on the evaluation
one. This is likely due to the default token level cross entropy loss not being
strong enough to reinforce the template filling task. To address this, we added a
component to the loss only looking at augmented parts of the gold label sentences,
since these corresponded to template role fillers. This loss component was weighted
with respect to the token level cross entropy loss. We found that higher weights for
the augmented spans improved recall at the expense of precision. This makes sense
since the weighting encourages the model to predict more augmentations.

## Multiple triggers per template
The number of examples generated per document is equal to the product of the numbers
of triggers per each template. More specifically, each example will have one of
the ordered tuples of trigger spans from the Cartesian Product over all the templates'
sets of trigger spans. More details can be found in [this PDF](triggers_formalized.pdf).

## Running experiments

### Setting up Conda environments
Different setup for running on GPUs depending on if GPU is the Ampere generation
or newer

#### Ampere and newer
1. `conda create -n TANL2 python=3`
2. `conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia`
2. In the folder for the specific environment, replace `trainer.train(model_path=model_args.model_name_or_path)`
with `trainer.train()`

#### Older than Ampere
1. `conda create -n TANL python=3.8`

### Running slurm files
1. `cd` into [`ampere+`](experiments/focused_cross_entropy/slurm/ampere+) or [`ampere-`](experiments/focused_cross_entropy/slurm/ampere-) for if you want to run with GPU newer or older than Ampere generation, respectively
2. Optionally change the GPU type or number of GPUs
3. Can also set other options; check [here](https://it.coecis.cornell.edu/researchit/g2cluster/)
4. `sbatch --requeue <slurm_file>.sub`

### Processing outputs and getting results
#### Plotting training loss
1. Copy [`logs.json`](experiments/focused_cross_entropy/g2_environments/all_argument_spans_2_trig/logs.json) into [`training_loss`](experiments/focused_cross_entropy/training_logs/training_loss/) directory
2. `python3 loss_pic.py <training_loss directory path>`

#### Computing training template level errors
1. Training loss: `python3 muc_errors.py --output_file ../experiments/<experiment type>/training_logs/training_muc_errors/<experiment name>.txt --is_eval False --model_outs ../experiments/<experiment type>/g2_environments/<experiment name>/train_predictions.txt --tanl_file ../original_scripts/data/mucevent/mucevent_dev.json --gtt_file ../../data/short_muc/gtt_dev.json`
2. Evaluation loss: `python3 muc_errors.py --output_file ../experiments/<experiment type>/training_logs/validation_muc_errors/<experiment name>.txt --is_eval True --model_outs ../experiments/<experiment type>/g2_environments/<experiment name>/train_predictions.txt --tanl_file ../original_scripts/data/mucevent/mucevent_test.json --gtt_file ../../data/short_muc/gtt_test.json`
3. Check [README](README.md) for debugging with `muc_errors.py`

#### Computing all test/validation errors
1. Transform outputs into Error Analaysis format: `python3 process_raw_outputs.py --model_outs ../experiments/<experiment type>/g2_environments/<experiment name>/test_predictions.txt --output_file ../experiments/<experiment type>/error_analysis_formatted/<experiment name>.out --tanl_file ../../data/full_eval/tanl_eval.json --gtt_file ../../data/full_eval/gtt_eval.json --debug_file process_raw_outputs_debug.json`
2. Check [README](README.md) for debugging with `process_raw_outputs.py`
3. `python3 Error_Analysis.py -i "../experiments/<experiment type>/error_analysis_formatted/<experiment name>.out" -o "../experiments/<experiment type>/error_analysis_outputs/<experiment name>.out" --verbose -s all -m "MUC_Errors" -at`
4. Check [Error Analysis repo](https://github.com/IceJinx33/auto-err-template-fill/tree/main) for more options for `Error_Analysis.py`