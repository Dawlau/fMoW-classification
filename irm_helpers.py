import torch
from tqdm import tqdm
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from SubsampledDataset import SubsampledDataset
import torch.autograd as autograd


LAMBDA = 1e-2
w = torch.tensor(1.).to("cuda").requires_grad_()


def compute_penalty(losses):
	grad_1 = autograd.grad(losses[0::2].mean(), w, create_graph=True)[0]
	grad_2 = autograd.grad(losses[1::2].mean(), w, create_graph=True)[0]
	return torch.sum(grad_1 * grad_2)


def train_step(model, data_loader, loss_fn, optimizer, device):
	model.train()

	accumulated_y_true = torch.tensor([]).to(device)
	accumulated_y_pred = torch.tensor([]).to(device)
	accumulated_metadata = torch.tensor([]).to(device)

	loss_value = 0

	for x, y, metadata in tqdm(data_loader):
		x = x.to(device)
		y = y.to(device)
		metadata = metadata.to(device)

		optimizer.zero_grad()

		y_pred = model(x)
		loss = loss_fn(y_pred * w, y)

		loss_mean = loss.mean()
		penalty = compute_penalty(loss)

		(LAMBDA * loss_mean + penalty).backward()
		optimizer.step()

		y_pred = torch.argmax(y_pred, dim=-1)

		accumulated_y_true = torch.concat(
			[accumulated_y_true, torch.flatten(y)], dim=0
		)
		accumulated_y_pred = torch.concat(
			[accumulated_y_pred, torch.flatten(y_pred)], dim=0
		)
		accumulated_metadata = torch.concat(
			[accumulated_metadata, metadata], dim=0
		)
		loss_value += loss_mean.item()
		print(loss_mean)

	accumulated_y_true = accumulated_y_true.cpu()
	accumulated_y_pred = accumulated_y_pred.cpu()
	accumulated_metadata = accumulated_metadata.cpu()
	loss_value = loss_value / len(data_loader)

	return accumulated_y_true, accumulated_y_pred, accumulated_metadata, loss_value


def val_step(model, data_loader, loss_fn, device):
	model.eval()

	accumulated_y_true = torch.tensor([]).to(device)
	accumulated_y_pred = torch.tensor([]).to(device)
	accumulated_metadata = torch.tensor([]).to(device)
	loss_value = 0

	with torch.no_grad():
		for x, y, metadata in tqdm(data_loader):
			x = x.to(device)
			y = y.to(device)
			metadata = metadata.to(device)

			y_pred = model(x)
			loss = loss_fn(y_pred, y)

			y_pred = torch.argmax(y_pred, dim=-1)

			accumulated_y_true = torch.concat(
				[accumulated_y_true, torch.flatten(y)], dim=0
			)
			accumulated_y_pred = torch.concat(
				[accumulated_y_pred, torch.flatten(y_pred)], dim=0
			)
			accumulated_metadata = torch.concat(
				[accumulated_metadata, metadata], dim=0
			)
			loss_value += loss.mean().item()

	accumulated_y_true = accumulated_y_true.cpu()
	accumulated_y_pred = accumulated_y_pred.cpu()
	accumulated_metadata = accumulated_metadata.cpu()
	loss_value = loss_value / len(data_loader)

	return accumulated_y_true, accumulated_y_pred, accumulated_metadata, loss_value


def build_metrics_dict(dataset, y_true, y_pred, metadata, loss):
	from SubsampledDataset import true_to_label

	metrics = dataset.eval(
		torch.tensor([true_to_label[y.item()] for y in y_true]), # convert to original labels
		torch.tensor([true_to_label[y.item()] for y in y_pred]),
		metadata
	)[0]
	metrics = {k: v for k, v in metrics.items() if "acc" in k and "year" not in k and ("Americas" in k or "Europe" in k)}
	metrics["loss"] = loss
	return metrics


def plot_graph(metric, train_evolution, val_evolution, id_val_evolution, NUM_EPOCHS):
	epochs = list(range(1, NUM_EPOCHS + 1))

	plt.plot(epochs, [x[metric] for x in train_evolution])
	plt.plot(epochs, [x[metric] for x in val_evolution])
	plt.plot(epochs, [x[metric] for x in id_val_evolution])
	plt.legend([f"train_{metric}", f"val_{metric}", f"id_val_{metric}"])
	plt.show()