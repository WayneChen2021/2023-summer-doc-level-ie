# Script usage info


## `empty_test_examples.py`
Generates empty TANL style examples and nonempty GTT style examples

### Flags
1. `muc_dir`: The directory containing the [original unprocessed MUC files](`../data/original_muc)
2. `tanl_out`: Output JSON file path of TANL style examples; only set if want TANL examples
3. `gtt_out`: Output JSON file path of GTT style examples; only set if want GTT examples


## `Error_Analysis.py`
Error Analysis script. Check usage in [original repo](https://github.com/IceJinx33/auto-err-template-fill/tree/main)


## `muc_errors.py`
Computes training time MUC template level errors for TANL

### Flags
1. `output_file`: Output txt file path; 1 JSON result per line
2. `is_eval`: Wether or not to compute errors on evaluation or training data; should
correspond with examples in `tanl_file` and `gtt_file`
3. `model_outs`: Output txt file ([example](../experiments/focused_cross_entropy/environments/all_argument_spans_2_trig/train_predictions.txt)) containing training time predictions; each output has a line of "trigger_output_sentence ..."
followed potentially by pairs of "predicted_relations ... \n gt_relations ... \n"
for each predicted event/template in the text (if any)
4. `tanl_file`: Path of JSON containing TANL style examples model was evaluating on;
could have no annotations, but need to have tokens; could generate using `empty_test_examples.py`
5. `gtt_file`: Path of JSON containing GTT style examples model was evaluating on;
must have all annotations; could generate using `empty_test_examples.py`
6. `edge_cases`: Path of JSON containing edge cases; JSON is of the form below,
where `<text_span>` is a section in the model output, `<trigger_span>` is the text
without brackets of the span labeled as a trigger, and `<trigger_type>` is one of
either "attack", "bombing", "kidnapping", "arson", "robbery", or "forced work stoppage";
is optional and defaults to 0
```JSON
{
    "<text_span1>": [["<trigger_span1>", "<trigger_type1>"], ["<trigger_span2>", "<trigger_type2>"]],
    "<text_span2>": [["<trigger_span3>", "<trigger_type3>"]]
}
```
7. `start_from`: Line in `model_outs` to start processing from; is optional
8. `error_analysis`: The [quick Error Analysis script](Error_Analysis_quick.py) file
path; is optional and defaults to `Error_Analysis_quick.py`


### Other notes
1. One common issue is that a single trigger is used for multiple templates. Our
regular expression parsing is not complex enough to handle these cases. To fix this,
put the edge case in the `edge_cases` file. The script will look for if `<text_span>`
is contained in the section after "trigger_output_sentence", and if so, will use
your provided trigger spans and types. One issue, however, is that the model may
correct itself such that the edge case may not be relevant for later predictions.
This means you may need to delete edge cases during running.
2. Less frequently, the number of predicted relationship sets will not match the
number of predicted triggers. This could be due to the above point, or some bug
in TANL's parsing of the generated augmented text. If this case, either delete or
add a relationship set in the `model_outs` file.
3. Save computation time by using `start_from`. As the script is running it will
report what line it got to (0-indexed) before running through a postprocessing file and the
`error_analysis` file. If a bug occurs, use the line count to find where the last
"EVAL PART" or "TEST PART" was and set it to that. The script will pick up reading
from that point.

## `process_og_muc.py`
Turns the original MUC files and trigger annotations into nonempty TANL and GTT style
examples

### Flags
1. `muc_dir`: The directory containing the [original unprocessed MUC files](`../data/original_muc)
2. `temp_trigs`: Max number of triggers per template; total number of TANL examples
generated per document is the product of the number of triggers per template
3. `annotation_file`: File of [trigger annotations](../../data/trigger_annotations/project-14-at-2023-06-29-05-20-eff93e87.json)
4. `tanl_train_out`: Output JSON of training TANL style examples; optionally set
if want this output
5. `tanl_eval_out`: Output JSON of evaluation (both test and dev) TANL style examples;
optionally set if want this output
6. `gtt_train_out`: Output JSON of training GTT style examples; optionally set
if want this output
7. `gtt_eval_out`: Output JSON of evaluation (both test and dev) GTT style examples;
optionally set if want this output

### Other notes
1. Unless you want to measure trigger prediction accuracy or have the entire evaluation
set labeled with triggers, `tanl_eval_out` and `gtt_eval_out` are probably not needed
2. `gtt_train_out` is probably not needed unless if you're computing MUC style errors
on the full training set
3. Check the comments in the code to make changes if you change the [schema](../experiments/focused_cross_entropy/environments/all_argument_spans_2_trig/data/mucevent_schema.json) or [types](../experiments/focused_cross_entropy/environments/all_argument_spans_2_trig/data/mucevent_types.json)
file, have more [labeled data](../../data/trigger_annotations/project-14-at-2023-06-29-05-20-eff93e87.json)
and need more edge cases, or want to use a different trigger selection mechanism.
The default trigger selection is the earliest k, and the other options are linked.

## `process_raw_outputs.py`
Processes predictions made by TANL into the format for the Error Analysis script

## Flags
1. `model_outs`: Output txt file ([example](../experiments/focused_cross_entropy/environments/all_argument_spans_2_trig/train_predictions.txt)) containing training time predictions; each output has a line of "trigger_output_sentence ..."
followed potentially by pairs of "predicted_relations ... \n gt_relations ... \n"
for each predicted event/template in the text (if any)
2. `output_file`: Output JSON file path for Error Analysis formatted outputs
3. `tanl_file`: See 4 in flags notes for `muc_errors.py`
4. `gtt_file`: See 5 in flags notes for `muc_errors.py`
5. `edge_cases`: See 6 in flags notes for `muc_errors.py`
6. `debug_file`: If a set of predictions has an unequal amount of triggers and predicted
templates, will log generated template to this JSON file; optionally set if you want
this logging; see other notes points 1, 2 in `muc_errors.py` for more info
7. `print_lines`: Will print each line and its line number; optionally set

## Other notes
1. May have to change `role map` (line 52) depending on what the augmented natural
language tags are (defined in [`mucevent_schema.json`](../experiments/focused_cross_entropy/environments/all_argument_spans_2_trig/data/mucevent_schema.json)
and [`mucevent_types.json`](../experiments/focused_cross_entropy/environments/all_argument_spans_2_trig/data/mucevent_types.json))
2. May have to change trigger regular expression (line 35) depending on how the
augmented natural language triggers are formatted (check trigger_output_sentence)

## `loss_pic.py`
Plots the training loss for TANL

## Flags
1. `log_dir`: Directory where model loss logs are stored ([example](../experiments/focused_cross_entropy/g2_environments/all_argument_spans_2_trig/train_predictions.txt))