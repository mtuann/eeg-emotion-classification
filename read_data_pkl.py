import pickle

fn = 'data_preprocessed_python/s01_processed.pkl'
data = pickle.load(open(fn, 'rb'), encoding='latin1')
print(data.keys())
print(data['labels'].shape)
print(data['data'].shape)