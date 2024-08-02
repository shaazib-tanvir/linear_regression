import india_gdp
import torch
import numpy as np
import matplotlib.pyplot as plt

from linear_regression import LinearRegression

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using Device: \033[32m{device.upper()}\033[0m")

    gdp_data = india_gdp.IndiaGDP("India_GDP_1960-2022.csv")
    input_tensor = torch.tensor([[gdp_data[i][0], 1] for i in range(15)], device=device, dtype=torch.float64)
    output_tensor = torch.tensor([[gdp_data[i][1], 1] for i in range(15)], device=device, dtype=torch.float64)

    lg = LinearRegression(input_tensor, output_tensor)
    lg.train()

    fig, ax = plt.subplots()
    ax.scatter([gdp_data[i][0] for i in range(20)], [gdp_data[i][1] for i in range(20)], c="red")

    x = np.linspace(2000, 2025, 50)
    y = [lg.get(torch.tensor([_x, 1], device=device))[0].item() for _x in x]
    ax.plot(x, y)

    plt.savefig("result.png")
