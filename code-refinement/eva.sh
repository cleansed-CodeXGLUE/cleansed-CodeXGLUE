#!/bin/sh

# # codebert eva small valid cleaned
# python evaluator/evaluator.py -ref ./data/small/valid-cleaned/inf-codebert/dev.gold -pre ./data/small/valid-cleaned/inf-codebert/dev.output
# python evaluator/CodeBLEU/calc_code_bleu.py --refs ./data/small/valid-cleaned/inf-codebert/dev.gold --hyp ./data/small/valid-cleaned/inf-codebert/dev.output --lang java --params 0.25,0.25,0.25,0.25


# # codebert eva small valid dirty
# python evaluator/evaluator.py -ref ./data/small/valid-org/inf-codebert/dev.gold -pre ./data/small/valid-org/inf-codebert/dev.output
# python evaluator/CodeBLEU/calc_code_bleu.py --refs ./data/small/valid-org/inf-codebert/dev.gold --hyp ./data/small/valid-org/inf-codebert/dev.output --lang java --params 0.25,0.25,0.25,0.25

# # codebert(clean) eva small valid(clean)
# python evaluator/evaluator.py -ref ./data/small/valid-cleaned/inf-codebert-clean-best/dev.gold -pre ./data/small/valid-cleaned/inf-codebert-clean-best/dev.output
# python evaluator/CodeBLEU/calc_code_bleu.py --refs ./data/small/valid-cleaned/inf-codebert-clean-best/dev.gold --hyp ./data/small/valid-cleaned/inf-codebert-clean-best/dev.output --lang java --params 0.25,0.25,0.25,0.25

# # codebert(clean) eva small valid(dirty)
python evaluator/evaluator.py -ref ./data/small/valid-org/inf-codebert-clean/dev.gold -pre ./data/small/valid-org/inf-codebert-clean/dev.output
python evaluator/CodeBLEU/calc_code_bleu.py --refs ./data/small/valid-org/inf-codebert-clean/dev.gold --hyp ./data/small/valid-org/inf-codebert-clean/dev.output --lang java --params 0.25,0.25,0.25,0.25

# lstm eva small valid cleaned
# python evaluator/evaluator.py -ref ./data/small/valid-cleaned/inf-lstm/dev.gold -pre ./data/small/valid-cleaned/inf-lstm/dev.output
# python evaluator/CodeBLEU/calc_code_bleu.py --refs ./data/small/valid-cleaned/inf-lstm/dev.gold --hyp ./data/small/valid-cleaned/inf-lstm/dev.output --lang java --params 0.25,0.25,0.25,0.25


# lstm(dirty) eva small valid dirty
# python evaluator/evaluator.py -ref ./data/small/valid-org/inf-lstm/dev.gold -pre ./data/small/valid-org/inf-lstm/dev.output
# python evaluator/CodeBLEU/calc_code_bleu.py --refs ./data/small/valid-org/inf-lstm/dev.gold --hyp ./data/small/valid-org/inf-lstm/dev.output --lang java --params 0.25,0.25,0.25,0.25

# lstm(clean) eva small valid(dirty)
# python evaluator/evaluator.py -ref ./data/small/valid-org/inf-lstm-clean/dev.gold -pre ./data/small/valid-org/inf-lstm-clean/dev.output
# python evaluator/CodeBLEU/calc_code_bleu.py --refs ./data/small/valid-org/inf-lstm-clean/dev.gold --hyp ./data/small/valid-org/inf-lstm-clean/dev.output --lang java --params 0.25,0.25,0.25,0.25

# lstm(clean) eva small valid(clean)
# python evaluator/evaluator.py -ref ./data/small/valid-cleaned/inf-lstm-clean/dev.gold -pre ./data/small/valid-cleaned/inf-lstm-clean/dev.output
# python evaluator/CodeBLEU/calc_code_bleu.py --refs ./data/small/valid-cleaned/inf-lstm-clean/dev.gold --hyp ./data/small/valid-cleaned/inf-lstm-clean/dev.output --lang java --params 0.25,0.25,0.25,0.25