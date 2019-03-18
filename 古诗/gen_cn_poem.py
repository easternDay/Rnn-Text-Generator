from textgenrnn import textgenrnn

textgen = textgenrnn(
  name="my.poem",
  weights_path='./my.poem_weights.hdf5',
  config_path='./my.poem_config.json',
  vocab_path='./my.poem_vocab.json'
)
textgen.generate(20, temperature=1.0)
