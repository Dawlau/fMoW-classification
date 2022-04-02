from torch.utils.data import Dataset
from wilds.common.grouper import CombinatorialGrouper
import torch


LABELS = set([42, 11, 36, 35, 29, 14, 30, 22, 54, 56, 48, 6, 24, 46, 47, 20, 3, 41, 32, 43])
REGIONS = set([
	1, # Americas
	3  # Europe
])
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class SubsampledDataset(Dataset):


	@staticmethod
	def filter_sample(sample, grouper):
		x, y, metadata = sample
		valid_label    = y.item() in LABELS
		valid_region   = grouper.metadata_to_group(torch.unsqueeze(metadata, dim=0)).item() in REGIONS
		return valid_label and valid_region


	def __init__(self, dataset, grouper):
		self.grouper = grouper
		self.dataset = dataset
		self.indices = [
			i for i, sample in enumerate(dataset) if SubsampledDataset.filter_sample(sample, self.grouper)
		]
		self.collate = None


	def __len__(self):
		return len(self.indices)


	def __getitem__(self, idx):
		return self.dataset[self.indices[idx]]