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

### experiments
#### config & hyper parameters
|hyper parameters|value|
|---|---|
|vocab_size|25000|
|seq_len|50|
|embed_dim|256|
|dense_dim|256|
|multi_head_num|8|
|conf_epoch|200|
|conf_batch_size|32|
|conf_optimizer|rmsprop|
|conf_activation|relu|
|conf_gen_len|25|


#### results
|prompts|generated lyric|
|---|---|
|'burn the witch'|'burn the witch fast turn days water darken promise blame wipe note world sand face think play turn father blame water sister cry cling conquer give tree forget'|
|'day dream'|'day dream toll player single cash wake fade morning season possession street refuse join even violent sweeter notice shout style closer like madness truths steal'|
|'dark deck'|'dark deck eye like fruit earn  cloud smooth gentle sleepy little bull somebody face luck luck lick story story story star paths fall go fall'|
|'desert island'|'desert island home hear tumble tough grind stone heart call stone night hear heart mask ways real whisper show silent go listen recall lover wall fuck deep'|
|'full stop'|'full stop know time time brand stop music friend hear dance brand different dance take play time walk good floor time play play simple game somebody play'|
|'glass eyes'|'glass eyes mirror folks hatch film evidence scene freeze word glue stupid punk pleasure stupid know stupid know stupid know stupid know stupid know story story'|
|'identity kit'|'identity kit pride sin eye time mean astray blind victim simple inside blue wrong wolves wrong gain sleep clue wrong pain find wrong kind find blind hard'|
|'the number'|'the number feelin live cool life want little yeah tough yeah trippin afraid learn guitars sing start guitar know fade gotta hard cover sing feelin perfection like'|
|'present tense'|'present tense ruin protect steam good powder sleep nervous cold arm rain angels warm cast cold fuel soul complain cold eye livin rain flow deal hair rain'|
|'tailor soldier sailor'|'tailor soldier sailor cars rhythm glow eye rack watch beat roll lips wine home bone tell change mind close eye certain sell head  cry texas'|
|'true love'|'true love things baby gonna share love cause forever sayin turn true go ordinary true true true things true true true true guitar blue play stay someday'|
