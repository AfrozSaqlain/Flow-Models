import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch
from torch._C import device
import torch.optim as optim
import tqdm
from sample import sampled_points
from fm import MLP
#from matplotlib.animation import FFMpegWriter
from matplotlib.animation import PillowWriter


device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

model = MLP(layers=5, channels=512).to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

data = torch.Tensor(sampled_points)
training_steps = 10_000
batch_size = 512
pbar = tqdm.tqdm(range(training_steps))
losses = []
for i in pbar:
    x1 = data[torch.randint(data.size(0), (batch_size,))]
    x0 = torch.randn_like(x1)
    target = (x1 - x0).to(device)
    t = torch.rand(x1.size(0))
    xt = (1 - t[:, None]) * x0 + t[:, None] * x1
    pred = model(xt.to(device), t.to(device))
    loss = ((target - pred)**2).mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    pbar.set_postfix(loss=loss.item())
    losses.append(loss.item())

plt.figure(figsize=(20, 8))
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel("Loss")
plt.grid()
plt.show()

# Inference
trajectory = []

xt = torch.randn(1000, 2).to(device)
steps = 200  # reduce for smoother video

for t in torch.linspace(0, 1, steps):
    trajectory.append(xt.detach().cpu().numpy())

    pred = model(xt, t.expand(xt.size(0)).to(device))
    xt = xt + (1 / steps) * pred



fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Left: fixed random samples
axes[0].scatter(trajectory[0][:, 0], trajectory[0][:, 1], s=5)
axes[0].set_title("Initial Noise")
axes[0].set_xlim(-4, 4)
axes[0].set_ylim(-4, 4)

# Right: evolving samples
scat = axes[1].scatter([], [], s=5)
axes[1].set_title("Flow Transformation")
axes[1].set_xlim(-4, 4)
axes[1].set_ylim(-4, 4)

def update(frame):
    data = trajectory[frame]
    scat.set_offsets(data)
    axes[1].set_title(f"Step {frame}")
    return scat,

# Add 3 seconds (30 fps * 3 sec = 90 frames) of the last frame to show complete transformation
total_frames = list(range(len(trajectory))) + [len(trajectory) - 1] * 90
anim = FuncAnimation(fig, update, frames=total_frames, interval=50)

anim.save("flow.gif", writer=PillowWriter(fps=30))
#anim.save("flow.mp4", writer=FFMpegWriter(fps=20))

plt.tight_layout()
plt.show()
