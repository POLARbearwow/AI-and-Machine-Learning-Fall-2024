# import matplotlib.pyplot as plt

# def plot_losses(losses, title="Model Loss Over Epochs"):
#     plt.figure()
#     for label, loss in losses.items():
#         plt.plot(loss, label=label)
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.title(title)
#     plt.legend()
#     plt.grid(True) 
#     plt.show()
# min-max & mean 与 none 差了几个数量级 同时绘制会导致前者重叠看不清且坐标轴不好看 因此分开绘制

import matplotlib.pyplot as plt

def plot_losses(losses, title="Model Loss Over Epochs"):
    fig, axs = plt.subplots(3, 1, figsize=(8, 12))  # 创建三个子图
    fig.suptitle(title)

    for ax, (label, loss) in zip(axs, losses.items()):
        ax.plot(loss, label=f'{label} Loss')
        ax.set_title(label)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
