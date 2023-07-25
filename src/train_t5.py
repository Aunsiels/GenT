# https://towardsdatascience.com/data-to-text-generation-with-t5-building-a-simple-yet-advanced-nlg-model-b5cce5a6df45

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, Adafactor

from sys import argv


train_file = argv[1]
test_file = argv[2]
num_of_epochs=3


def preprocess(filename):
    res = []
    with open(filename) as f:
        for line in f:
            line = line.strip().replace("<|endoftext|>", "").split("[SEP]")
            res.append(line)
    return res


training_data = preprocess(train_file)
testing_data = preprocess(test_file)


batch_size=8
num_of_batches=len(training_data)//batch_size

if torch.cuda.is_available():
   dev = torch.device("cuda:0")
   print("Running on the GPU")
else:
   dev = torch.device("cpu")
   print("Running on the CPU")


tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("t5-base",
                                                   return_dict=True)
#moving the model to GPU
model.to(dev)

optimizer = Adafactor(model.parameters(),lr=1e-3,
                      eps=(1e-30, 1e-3),
                      clip_threshold=1.0,
                      decay_rate=-0.8,
                      beta1=None,
                      weight_decay=0.0,
                      relative_step=False,
                      scale_parameter=False,
                      warmup_init=False)


#Sets the module in training mode
model.train()

loss_per_10_steps=[]
for epoch in range(1, num_of_epochs+1):
  print('Running epoch: {}'.format(epoch))

  running_loss = 0

  for i in range(num_of_batches):
    inputbatch = []
    labelbatch = []
    batch = training_data[i*batch_size:i*batch_size+batch_size]
    for line in batch:
      input = 'OPENKB: '+ line[0] +'</s>'
      labels = line[1] +'</s>'
      inputbatch.append(input)
      labelbatch.append(labels)
    inputbatch=tokenizer.batch_encode_plus(inputbatch,padding=True,max_length=400,return_tensors='pt')["input_ids"]
    labelbatch=tokenizer.batch_encode_plus(labelbatch,padding=True,max_length=400,return_tensors="pt") ["input_ids"]
    inputbatch=inputbatch.to(dev)
    labelbatch=labelbatch.to(dev)

    # clear out the gradients of all Variables
    optimizer.zero_grad()

    # Forward propogation
    outputs = model(input_ids=inputbatch, labels=labelbatch)
    loss = outputs.loss
    loss_num=loss.item()
    logits = outputs.logits
    running_loss+=loss_num
    if i%10 ==0:
      loss_per_10_steps.append(loss_num)

    # calculating the gradients
    loss.backward()

    #updating the params
    optimizer.step()

  running_loss=running_loss/int(num_of_batches)
  print('Epoch: {} , Running loss: {}'.format(epoch,running_loss))


torch.save(model.state_dict(), argv[3])
