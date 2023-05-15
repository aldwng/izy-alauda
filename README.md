# izy-alauda
Easy implementations of the model Transformer.
A playground of transformer applications.

## toy .00
### directory
/alauda/egg/tran_dec_gen_00.py

### introduction

This toy project is based on a small-scale public lyrics data set, and uses the Decoder of the Transformer structure to apply self-supervised training on the lyrics dataset. And try to use the trained model for lyrics generation. To observe the generated results according to several given prompts.

### modeling
dataset: open lyrics data downloaded from Kaggle\
input(decoder output): lyric lines with the last word masked\
output(transformer output): lyric lines with the first word masked
|prompts|
|---|
|'burn the witch'|
|'day dream'|
|'dark deck'|
|'desert island'|
|'full stop'|
|'glass eyes'|
|'identity kit'|
|'the number'|
|'present tense'|
|'tailor soldier sailor'|
|'true love'|

### experiments
#### config & hyper parameters

#### results
