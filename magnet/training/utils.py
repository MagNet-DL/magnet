import torch, pickle

def load_state(module, path, alternative_name=None):
	name = alternative_name if not hasattr(module, 'name') else module.name
	if name is None: raise RuntimeError('Module Name is None!')

	filepath = path / (name + '.pt')
	if filepath.exists(): module.load_state_dict(torch.load(filepath))

def save_state(module, path, alternative_name=None):
	name = alternative_name if not hasattr(module, 'name') else module.name
	if name is None: raise RuntimeError('Module Name is None!')

	path.mkdir(parents=True, exist_ok=True)
	filepath = path / (name + '.pt')

	torch.save(module.state_dict(), filepath)

def load_object(path, **kwargs):
	if path.exists():
		with open(path, 'rb') as f: return pickle.load(f)
	elif 'default' in kwargs.keys():
		return kwargs['default']
	else:
		raise RuntimeError(f'The path {path} does not exist. No default provided either.')

def save_object(obj, path):
	path.parent.mkdir(parents=True, exist_ok=True)
	with open(path, 'wb') as f: pickle.dump(obj, f)