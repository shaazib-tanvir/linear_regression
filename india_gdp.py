from torch.utils.data import Dataset
import pandas as pd

class IndiaGDP(Dataset):
    def __init__(self, file_name: str):
        self._data = pd.read_csv(file_name, header=1)

    def __len__(self):
        return len(self._data["Year"])

    def __getitem__(self, index):
        year = self._data["Year"][index]
        gdp = self._data["GDP in (Billion) $"][index]

        return year, gdp
