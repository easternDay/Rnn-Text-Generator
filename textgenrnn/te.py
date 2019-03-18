from textgenrnn import textgenrnn
textgen = textgenrnn()
#textgen.generate(5)
#生成五段文字

#generated_texts = textgen.generate(n=5, prefix="Trump", temperature=0.2, return_as_list=True)
#print(generated_texts)
#生成以Trump开头的五段文字

texts = ['Never gonna give you up, never gonna let you down',
            'Never gonna run around and desert you',
            'Never gonna make you cry, never gonna say goodbye',
            'Never gonna tell a lie and hurt you']

textgen.train_on_texts(texts, num_epochs=2,  gen_epochs=2)

textgen.generate_to_file('textgenrnn_texts.txt', n=5)
#将结果输出到文本

'''
textgen = textgenrnn(name="chinese_poetry")
textgen.reset()
textgen.train_from_file('./datasets/chinese-poetry.txt',
                        new_model=True,
                        batch_size=4,
                        rnn_bidirectional=True,
                        rnn_size=64,
                        dim_embeddings=300,
                        num_epochs=20)

print(textgen.model.summary())
'''

'''
textgen = textgenrnn(
  name="chinese_poetry",
  weights_path='./chinese_poetry_weights.hdf5',
  config_path='./chinese_poetry_config.json',
  vocab_path='./chinese_poetry_vocab.json'
)
textgen.generate(20, temperature=1.0)

'''

'''
textgen = textgenrnn(name="shakespeare")
textgen.reset()
textgen.train_from_file('./datasets/shakespeare.txt',
                        new_model=True,
                        batch_size=4,
                        rnn_bidirectional=True,
                        rnn_size=64,
                        dim_embeddings=300,
                        num_epochs=20)

print(textgen.model.summary())

'''
