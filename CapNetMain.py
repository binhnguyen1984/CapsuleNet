from CapNet import CapNet

model = CapNet( n_classes = 10, image_size = 28, n_hiddens = [512, 1024])
model.train()