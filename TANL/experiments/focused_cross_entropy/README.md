# Overview of experiments
1. all_argument_spans_2_trig: Add cross entropy loss component to anywhere in the
gold label sentence or the model generated sentence enclosed in a bracket. Also
add CE loss component to anywhere with a type (excluding "template entity"). All
loss components weighted the same. Max 2 triggers per template.
2. all_argument_spans_3_trig: Add cross entropy loss component to anywhere in the
gold label sentence or the model generated sentence enclosed in a bracket. Also
add CE loss component to anywhere with a type (excluding "template entity"). All
loss components weighted the same. Max 3 triggers per template.
3. all_argument_spans_2_trig_no_type: Add cross entropy loss component to anywhere in the
gold label sentence or the model generated sentence enclosed in a bracket. All
loss components weighted the same. Max 2 triggers per template.
4. all_argument_spans_2_trig_no_type: Add cross entropy loss component to anywhere in the
gold label sentence or the model generated sentence enclosed in a bracket. All
loss components weighted the same. Max 3 triggers per template.
5. bracket_and_types_2_trig: Add cross entropy loss component to anywhere in the
gold label sentence enclosed in a bracket. Also add CE loss component to anywhere
with a type (excluding "template entity"). All loss components weighted the same.
Max 2 triggers per template.
6. bracket_and_types_3_trig: Add cross entropy loss component to anywhere in the
gold label sentence enclosed in a bracket. Also add CE loss component to anywhere
with a type (excluding "template entity"). All loss components weighted the same.
Max 3 triggers per template.
7. bracket_only_2_trig: Add cross entropy loss component to anywhere in the
gold label sentence enclosed in a bracket. All loss components weighted the same.
Max 2 triggers per template.
8. bracket_only_2_trig: Add cross entropy loss component to anywhere in the
gold label sentence enclosed in a bracket. All loss components weighted the same.
Max 3 triggers per template.
9. heavy_weight_10: Add cross entropy loss component to anywhere in the
gold label sentence enclosed in a bracket. The extra component is weighted 10 times
greater than the cross entropy loss. Max 2 triggers per template.
10. heavy_weight_100: Add cross entropy loss component to anywhere in the
gold label sentence enclosed in a bracket. The extra component is weighted 100 times
greater than the cross entropy loss. Max 2 triggers per template.
11. negative_loss_1: Add cross entropy loss component to anywhere in the
gold label sentence enclosed in a bracket. Also adds negative loss components to
anywhere in the gold label sentence without a bracket (split into 2 losses for left
and right brackets). All loss components weighted the same. Max 2 triggers per template.
12. negative_loss_01: Add cross entropy loss component to anywhere in the
gold label sentence enclosed in a bracket. Also adds negative loss components to
anywhere in the gold label sentence without a bracket (split into 2 losses for left
and right brackets). The negative losses are weighted by 0.1. Max 2 triggers per template.