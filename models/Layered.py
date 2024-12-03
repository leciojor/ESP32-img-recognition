import numpy as np
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import is_available
import torch




#New architecture based on arch from option 1

class CNNModelForLayering(nn.Module):
    def __init__(self, mini=True):
        super(CNNModelForLayering, self).__init__()
        self.mini = mini

        #First convulutional layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        #Second convulutional layer
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.AdaptiveMaxPool2d((1, None))

        # Fully connected layers
        if self.mini:
          self.fc1 = nn.Linear(32 * 1 * 64, 512)  
        else:
          self.fc1 = nn.Linear(64 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, 128)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class finalConnectedNetwork(nn.Module):

  @staticmethod
  def cluster_objs(batch, k, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    # Object Detection Logic
    final_imgs = []
    for img in batch:
      if is_available():
        img = img.cpu()
      image = img.numpy()
      image = np.array(image, dtype=np.float64) / 255
      rows, cols, channels = image.shape
      image_2d = image.reshape(-1, channels)

      is_white = np.all(image_2d == 1.0, axis=1)
      non_white_pixels = image_2d[~is_white]
      white_positions = np.where(is_white)[0]

      kmeans = KMeans(n_clusters=k, random_state=42)
      available_samples = len(non_white_pixels)
      sample_size = min(1000, available_samples)  
      if sample_size > 0:
          image_sample = shuffle(non_white_pixels, random_state=42, n_samples=sample_size)
          kmeans.fit(image_sample)
      else:
          raise ValueError("You cannot train this network with a fully white image")

      labels = kmeans.predict(non_white_pixels)

      full_labels = np.full(image_2d.shape[0], -1)
      full_labels[~is_white] = labels

    
      # Generate grayscale images for each cluster
      sequence = []
      for cluster in range(k):
          grayscale_image = np.where(full_labels.reshape(rows, cols) == cluster, 0, 1)
          sequence.append(grayscale_image[np.newaxis, :, :]) 
      final_imgs.append(np.stack(sequence))  

    return torch.tensor(np.stack(final_imgs), dtype=torch.float32).to(device)


  def __init__(self, k):
        super(finalConnectedNetwork, self).__init__()
        self.k = k
        self.models = nn.ModuleList([CNNModelForLayering().to("cuda") if is_available() else CNNModelForLayering() for _ in range(k)])
        self.shared_fc = nn.Linear(k * 128, 4)

  def forward(self, batch_input):
      inputs = self.cluster_objs(batch_input, self.k)  # Shape: [batch_size, k, 1, height, width]
      batch_size = inputs.size(0)
      features = []
      for i, model in enumerate(self.models):
          cluster_inputs = inputs[:, i, :, :, :]  # Shape: [batch_size, 1, height, width]
          cluster_features = model(cluster_inputs)
          features.append(cluster_features)

      combined = torch.cat(features, dim=1)  # Shape: [batch_size, k * 128]

      # Shared fully connected layer
      output = self.shared_fc(combined)  # Shape: [batch_size, 4]
      return output


def weighted_voting(probabilities, weights):
    """
    Ensemble predictions using weighted voting.

    Args:
    - probabilities: A 3D array where each element is a probability vector (n_models x n_samples x n_classes).
    - weights: A list of weights corresponding to each model.

    Returns:
    - Final ensemble prediction (class with highest weighted probability).
    """
    weighted_probabilities = np.tensordot(weights, probabilities, axes=1)
    return np.argmax(weighted_probabilities, axis=1)

