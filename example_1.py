

import torch
import torch.nn.functional as F
import numpy
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1), is considered as the batch size of 100
y = x.pow(2) + 0.2*torch.rand(x.size())                 # noisy y data (tensor), shape=(100, 1)

# # print out the input and result
# plt.scatter(x.numpy(), y.numpy())
# plt.show()

# --------- build our network ------------
class Net(torch.nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_input, n_hidden)   # hidden layer
        self.out = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))  # activation function for hidden layer
        x = self.out(x)             # linear output
        return x

model = Net(1, 10, 1)
print(model)

# an alternative way to build your network
# model2 = torch.nn.Sequential(
#     torch.nn.Linear(1, 10),
#     torch.nn.ReLU(),
#     torch.nn.Linear(10, 1)
# )

# Here we need to choose an optimization method to update our weights
# pytorch contains many popular optimizers such as SGD, momentum, Adagrad, RMSprop, Adam, etc.
# Or we can write our own method manually.
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

# The nn package also contains definitions of popular loss functions; in this
# case we will use Mean Squared Error (MSE) as our loss function.
# Since MSE is good enough to handle the simple regression problem.
loss_func = torch.nn.MSELoss()


# Turn interactive mode on. this is the call to matplotlib that allows dynamic plotting
plt.ion()

for epoch in range(100):
    prediction = model(x)

    loss = loss_func(prediction, y)     # calculate loss

    # ---- method 1: using optimizer ----
    optimizer.zero_grad()               # reset all gradients
    loss.backward()                     # This function will compute gradients for all learnable parameters in the model and update the gradients.
    optimizer.step()                    # update the weights based on gradients and lr

    # ---- method 2: manually define your own update method ----
    # model.zero_grad()
    # loss.backward()
    # with torch.no_grad():
    #     for param in model.parameters():
    #         param -= learning_rate * param.grad

    # ------- This part is only about plotting ----------
    if epoch % 5 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()


