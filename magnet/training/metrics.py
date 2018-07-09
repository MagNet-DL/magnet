def accuracy(scores, y):
	y_pred = scores.max(1)[1]
	return (y_pred == y).float().mean()