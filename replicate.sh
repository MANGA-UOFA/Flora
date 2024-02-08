## Gradient Accumulation (T5-3B)

### Baseline (tuned learning rate)

python enc_dec_s2s.py optimizer=adafactor optimizer.learning_rate=4e-4 model.pretrained=true model.model_name_or_path=t5-3b grad_acc.steps=16 training.per_device_train_batch_size=1 training.eval_steps=1250

### LoRA (tuned learning rate)
python enc_dec_s2s.py optimizer=adafactor optimizer.learning_rate=1e-3 model.pretrained=true model.model_name_or_path=t5-3b grad_acc.steps=16 training.per_device_train_batch_size=1 training.eval_steps=1250 lora.disabled=false lora.tune_others=true lora.rank=256

### Flora (reusing the learning rate from the baseline)
python enc_dec_s2s.py optimizer=adafactor optimizer.learning_rate=4e-4 model.pretrained=true model.model_name_or_path=t5-3b grad_acc.steps=16 training.per_device_train_batch_size=1 training.eval_steps=1250 grad_acc.impl=compressed grad_acc.tau=256


## Momentum (T5-small)
python enc_dec_s2s.py optimizer=adafactor model.pretrained=false model.model_name_or_path=t5-small optimizer.learning_rate=1e-3  training.per_device_train_batch_size=4 training.eval_steps=200000 optimizer.momentum=0.9  training.num_train_epochs=10

### LoRA (tuned learning rate)
python enc_dec_s2s.py optimizer=adafactor model.pretrained=false model.model_name_or_path=t5-small optimizer.learning_rate=3e-3  training.per_device_train_batch_size=4 training.eval_steps=200000 training.num_train_epochs=10 optimizer.momentum=0.9 lora.disabled=false lora.tune_others=true lora.rank=256

### Flora (reusing the learning rate from the baseline)
python enc_dec_s2s.py optimizer=flora model.pretrained=false model.model_name_or_path=t5-small optimizer.learning_rate=1e-3  training.per_device_train_batch_size=4 training.eval_steps=200000 optimizer.b1=0.9  training.num_train_epochs=10 optimizer.tau=256 optimizer.kappa=1000 
