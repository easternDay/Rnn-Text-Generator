from textgenrnn import textgenrnn

textgen = textgenrnn(
  name="expl",
  weights_path='./expl_weights.hdf5',
  config_path='./expl_config.json',
  vocab_path='./expl_vocab.json'
)
textgen.generate(20, temperature=1.0)
