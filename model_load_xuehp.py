# -*- coding: utf-8 -*-
# author: huihui
# date: 2020/4/2 9:28 上午 
# coding: utf-8
from sogou_mrc.data.vocabulary import Vocabulary
from sogou_mrc.dataset.squad import SquadReader, SquadEvaluator
from sogou_mrc.model.bidaf import BiDAF
import tensorflow as tf
import logging
from sogou_mrc.data.batch_generator import BatchGenerator
from sogou_mrc.train.trainer import Trainer

tf.logging.set_verbosity(tf.logging.ERROR)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

print('load data')
logging.info('load data')
data_folder = '/root/SogouMRCToolkit/data/'
dev_file = data_folder + "dev-v1.1.json"

reader = SquadReader()
eval_data = reader.read(dev_file)
evaluator = SquadEvaluator(dev_file)

print('load vocab')
logging.info('load vocab')
vocab = Vocabulary()
vocab_save_path='/root/SogouMRCToolkit/data/vocab.json'
vocab.load(vocab_save_path) # load vocab from save path

word_embedding = vocab.make_word_embedding('/root/SogouMRCToolkit/data/' + "glove.6B.100d.txt")

test_batch_generator = BatchGenerator(vocab, eval_data, batch_size=60)

print('load model')
logging.info('load mocel')
save_dir='/root/SogouMRCToolkit/data/best_weights'+'/best_weights'
model = BiDAF(vocab,pretrained_word_embedding=word_embedding)
model.load(save_dir)
model.inference(test_batch_generator) # inference on test data

logging.info('compile')
print('compile')
model.compile(tf.train.AdamOptimizer, 0.001)

print('session')
model.session.run(tf.local_variables_initializer())

#model.evaluate(test_batch_generator,evaluator)

print('evaluate')


#logging.info("evaluate")
#model.evaluate(test_batch_generator,evaluator)


eval_batch_generator = test_batch_generator
eval_batch_generator.init()
eval_instances = eval_batch_generator.get_instances()
model.session.run(model.eval_metric_init_op)

eval_num_steps = (eval_batch_generator.get_instance_size() + eval_batch_generator.get_batch_size() - 1) // eval_batch_generator.get_batch_size()
output = Trainer._eval_sess(model, eval_batch_generator, eval_num_steps, None)
pred_answer = model.get_best_answer(output, eval_instances)
print('pred_answer={}'.format(pred_answer))
score = evaluator.get_score(model.get_best_answer(output, eval_instances))

metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in score.items())
print("- Eval metrics: " + metrics_string)



#print(SquadEvaluator.exact_match_score())
#print(SquadEvaluator.f1_score())

