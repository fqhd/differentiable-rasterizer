import matplotlib.pyplot as plt

with open('losses.txt') as f:
    losses = f.read().split('\n')[:-1] # Discard last item because it's empty

losses = [float(a) for a in losses]

plt.plot(losses)
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.show()
