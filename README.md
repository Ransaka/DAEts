# Time series clustering using KMeans and DAE (Denoising Auto Encoders)

## Problem definition

![problem_definition.png](images/problem_definition.png)

Say we have 10,000 customer's usage data for past 500 Days. And we need to build usage forecasting model for this 10,000 users. But in some cases all usaers don't nessasaryly want machine learning methods. Instead we can use simple rule for some customer segment. But the situation is quite complex in some customer segments, their usage is hard to predict with simple rules, in such a case we have to use machine learning. But how do we distinguish those customer segments? That's where segmentation comes into a play. 

In this notebook. we will build time series clustering model to identify above mentioned customer segments. Dataset used here represent the real customer usage dataset extracted between 2020/2021. Let's begin.

```python
from TabDAE.tab_dae import DenoisingAutoencoder

import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler

import warnings

warnings.filterwarnings('ignore')
```

```python
usage_data = pd.read_parquet("usage_data/sample_10_000_preprocessed.parquet")
```


```python
usage_data.columns = ['uid','usage']
```


```python
usage_data_pivot = usage_data.pivot_table(
    values='usage',
    index='uid',
    columns=usage_data.index
)
```


```python
usage_data_pivot.shape
```




    (10000, 579)




```python
usage_data_pivot.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>2020-06-01</th>
      <th>2020-06-02</th>
      <th>2020-06-03</th>
      <th>2020-06-04</th>
      <th>2020-06-05</th>
      <th>2020-06-06</th>
      <th>2020-06-07</th>
      <th>2020-06-08</th>
      <th>2020-06-09</th>
      <th>2020-06-10</th>
      <th>...</th>
      <th>2021-12-22</th>
      <th>2021-12-23</th>
      <th>2021-12-24</th>
      <th>2021-12-25</th>
      <th>2021-12-26</th>
      <th>2021-12-27</th>
      <th>2021-12-28</th>
      <th>2021-12-29</th>
      <th>2021-12-30</th>
      <th>2021-12-31</th>
    </tr>
    <tr>
      <th>uid</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>255</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>176</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1100</td>
      <td>1726</td>
      <td>1243</td>
      <td>1426</td>
      <td>1366</td>
      <td>1215</td>
      <td>859</td>
      <td>1772</td>
      <td>0</td>
      <td>1099</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>341</td>
      <td>647</td>
      <td>634</td>
      <td>46</td>
      <td>32</td>
      <td>284</td>
      <td>374</td>
      <td>451</td>
      <td>0</td>
      <td>1006</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>304</td>
      <td>0</td>
      <td>323</td>
      <td>88</td>
      <td>22</td>
      <td>295</td>
      <td>413</td>
      <td>392</td>
      <td>0</td>
      <td>114</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 579 columns</p>
</div>




```python
usage_data_pivot.columns = list(
    range(
        len(
            usage_data_pivot.columns
            )
        )
    )
```


```python
del usage_data
```


```python
usage_data_pivot.replace(0,1e-4,inplace=True)
usage_data_pivot.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>569</th>
      <th>570</th>
      <th>571</th>
      <th>572</th>
      <th>573</th>
      <th>574</th>
      <th>575</th>
      <th>576</th>
      <th>577</th>
      <th>578</th>
    </tr>
    <tr>
      <th>uid</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0001</td>
      <td>0.0001</td>
      <td>0.0001</td>
      <td>0.0001</td>
      <td>0.0001</td>
      <td>0.0001</td>
      <td>0.0001</td>
      <td>0.0001</td>
      <td>0.0001</td>
      <td>0.0001</td>
      <td>...</td>
      <td>0.0001</td>
      <td>0.0001</td>
      <td>0.0001</td>
      <td>0.0001</td>
      <td>0.0001</td>
      <td>0.0001</td>
      <td>0.0001</td>
      <td>0.0001</td>
      <td>0.0001</td>
      <td>0.0001</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0001</td>
      <td>0.0001</td>
      <td>0.0001</td>
      <td>0.0001</td>
      <td>0.0001</td>
      <td>0.0001</td>
      <td>0.0001</td>
      <td>0.0001</td>
      <td>0.0001</td>
      <td>0.0001</td>
      <td>...</td>
      <td>0.0001</td>
      <td>255.0000</td>
      <td>0.0001</td>
      <td>0.0001</td>
      <td>0.0001</td>
      <td>0.0001</td>
      <td>0.0001</td>
      <td>176.0000</td>
      <td>0.0001</td>
      <td>0.0001</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0001</td>
      <td>0.0001</td>
      <td>0.0001</td>
      <td>0.0001</td>
      <td>0.0001</td>
      <td>0.0001</td>
      <td>0.0001</td>
      <td>0.0001</td>
      <td>0.0001</td>
      <td>0.0001</td>
      <td>...</td>
      <td>1100.0000</td>
      <td>1726.0000</td>
      <td>1243.0000</td>
      <td>1426.0000</td>
      <td>1366.0000</td>
      <td>1215.0000</td>
      <td>859.0000</td>
      <td>1772.0000</td>
      <td>0.0001</td>
      <td>1099.0000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0001</td>
      <td>0.0001</td>
      <td>0.0001</td>
      <td>0.0001</td>
      <td>0.0001</td>
      <td>0.0001</td>
      <td>0.0001</td>
      <td>0.0001</td>
      <td>0.0001</td>
      <td>0.0001</td>
      <td>...</td>
      <td>341.0000</td>
      <td>647.0000</td>
      <td>634.0000</td>
      <td>46.0000</td>
      <td>32.0000</td>
      <td>284.0000</td>
      <td>374.0000</td>
      <td>451.0000</td>
      <td>0.0001</td>
      <td>1006.0000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0001</td>
      <td>0.0001</td>
      <td>0.0001</td>
      <td>0.0001</td>
      <td>0.0001</td>
      <td>0.0001</td>
      <td>0.0001</td>
      <td>0.0001</td>
      <td>0.0001</td>
      <td>0.0001</td>
      <td>...</td>
      <td>304.0000</td>
      <td>0.0001</td>
      <td>323.0000</td>
      <td>88.0000</td>
      <td>22.0000</td>
      <td>295.0000</td>
      <td>413.0000</td>
      <td>392.0000</td>
      <td>0.0001</td>
      <td>114.0000</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 579 columns</p>
</div>



Here I'm deviding each users's usage by his/her maximum usage in given 579 days. Now I have 0-1 normalized value for each user. Since my target is identifying users with similar usage pattern (not the usage quantity) I guess it'll be more meaningful.


```python
normalize = lambda row: (row / row.max())
```


```python
usage_normalized = usage_data_pivot.apply(normalize, axis=1)
```


```python
usage_normalized.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>569</th>
      <th>570</th>
      <th>571</th>
      <th>572</th>
      <th>573</th>
      <th>574</th>
      <th>575</th>
      <th>576</th>
      <th>577</th>
      <th>578</th>
    </tr>
    <tr>
      <th>uid</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3.885004e-08</td>
      <td>3.885004e-08</td>
      <td>3.885004e-08</td>
      <td>3.885004e-08</td>
      <td>3.885004e-08</td>
      <td>3.885004e-08</td>
      <td>3.885004e-08</td>
      <td>3.885004e-08</td>
      <td>3.885004e-08</td>
      <td>3.885004e-08</td>
      <td>...</td>
      <td>3.885004e-08</td>
      <td>3.885004e-08</td>
      <td>3.885004e-08</td>
      <td>3.885004e-08</td>
      <td>3.885004e-08</td>
      <td>3.885004e-08</td>
      <td>3.885004e-08</td>
      <td>3.885004e-08</td>
      <td>3.885004e-08</td>
      <td>3.885004e-08</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.524615e-08</td>
      <td>2.524615e-08</td>
      <td>2.524615e-08</td>
      <td>2.524615e-08</td>
      <td>2.524615e-08</td>
      <td>2.524615e-08</td>
      <td>2.524615e-08</td>
      <td>2.524615e-08</td>
      <td>2.524615e-08</td>
      <td>2.524615e-08</td>
      <td>...</td>
      <td>2.524615e-08</td>
      <td>6.437768e-02</td>
      <td>2.524615e-08</td>
      <td>2.524615e-08</td>
      <td>2.524615e-08</td>
      <td>2.524615e-08</td>
      <td>2.524615e-08</td>
      <td>4.443322e-02</td>
      <td>2.524615e-08</td>
      <td>2.524615e-08</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.759944e-08</td>
      <td>1.759944e-08</td>
      <td>1.759944e-08</td>
      <td>1.759944e-08</td>
      <td>1.759944e-08</td>
      <td>1.759944e-08</td>
      <td>1.759944e-08</td>
      <td>1.759944e-08</td>
      <td>1.759944e-08</td>
      <td>1.759944e-08</td>
      <td>...</td>
      <td>1.935938e-01</td>
      <td>3.037663e-01</td>
      <td>2.187610e-01</td>
      <td>2.509680e-01</td>
      <td>2.404083e-01</td>
      <td>2.138332e-01</td>
      <td>1.511792e-01</td>
      <td>3.118620e-01</td>
      <td>1.759944e-08</td>
      <td>1.934178e-01</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.786671e-08</td>
      <td>1.786671e-08</td>
      <td>1.786671e-08</td>
      <td>1.786671e-08</td>
      <td>1.786671e-08</td>
      <td>1.786671e-08</td>
      <td>1.786671e-08</td>
      <td>1.786671e-08</td>
      <td>1.786671e-08</td>
      <td>1.786671e-08</td>
      <td>...</td>
      <td>6.092550e-02</td>
      <td>1.155976e-01</td>
      <td>1.132750e-01</td>
      <td>8.218689e-03</td>
      <td>5.717349e-03</td>
      <td>5.074147e-02</td>
      <td>6.682151e-02</td>
      <td>8.057888e-02</td>
      <td>1.786671e-08</td>
      <td>1.797391e-01</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.980198e-08</td>
      <td>1.980198e-08</td>
      <td>1.980198e-08</td>
      <td>1.980198e-08</td>
      <td>1.980198e-08</td>
      <td>1.980198e-08</td>
      <td>1.980198e-08</td>
      <td>1.980198e-08</td>
      <td>1.980198e-08</td>
      <td>1.980198e-08</td>
      <td>...</td>
      <td>6.019802e-02</td>
      <td>1.980198e-08</td>
      <td>6.396040e-02</td>
      <td>1.742574e-02</td>
      <td>4.356436e-03</td>
      <td>5.841584e-02</td>
      <td>8.178218e-02</td>
      <td>7.762376e-02</td>
      <td>1.980198e-08</td>
      <td>2.257426e-02</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 579 columns</p>
</div>



Plot sample usage pattern


```python
usage_normalized.iloc[999].plot(alpha=0.5,color='r')
plt.grid(True)
plt.show()
```


    
![png](README_files/README_17_0.png)
    


Let's train the model. Please note the output layer activation function used here. It's `RelU (Rectified Linear Unit)` . I used it because customer usage data always be either 0 or positive. If we use another activation functions such as `Logistic` it may return negetive values and it's meaning-less for this specific problem. If you want to know more info about model architecture, please refre to [this](https://github.com/Ransaka/TabDAE) code repo. It containes all the infomation related to above model archtecture. Finally I have used `MAE` as loss function here.


```python
# Split the data into train and test sets
train_data, test_data = train_test_split(usage_normalized, test_size=0.2)

# Convert the data to a PyTorch tensor
train_data = torch.tensor(train_data.values).float()
test_data = torch.tensor(test_data.values).float()

# Define the model, optimizer, and loss function
input_dim = usage_normalized.shape[1]
encoding_dim = 64
hidden_dims = [512 ,256, 128]
model = DenoisingAutoencoder(input_dim, hidden_dims, encoding_dim,dropout_rate=0.1,activation=nn.ReLU,output_activation=nn.ReLU)
optimizer = optim.Adam(model.parameters(),lr=1e-2)
loss_fn = nn.L1Loss()

# Train the model
batch_size = 5000
num_epochs = 20
for epoch in range(num_epochs):
    # Iterate over the training data in batches
    for i in range(0, len(train_data), batch_size):
        inputs = train_data[i:i+batch_size]
        # Forward pass
        output = model(inputs, add_noise=True)
        loss = loss_fn(output, inputs)
        
        # Backward pass and optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Evaluate the model on the test set
    model.eval()
    
    test_loss = 0
    for i in range(0, len(test_data), batch_size):
        inputs = test_data[i:i+batch_size]
        output = model(inputs, add_noise=False)
        test_loss += loss_fn(output, inputs).item()
    test_loss /= len(test_data)
    print(f'Test loss: {test_loss:.5f}')
    
# switch back to train mode    
    model.train()
```

    Test loss: 0.00008
    Test loss: 0.00017
    Test loss: 0.00019
    Test loss: 0.00028
    Test loss: 0.00029
    Test loss: 0.00027
    Test loss: 0.00021
    Test loss: 0.00017
    Test loss: 0.00013
    Test loss: 0.00010
    Test loss: 0.00009
    Test loss: 0.00007
    Test loss: 0.00006
    Test loss: 0.00006
    Test loss: 0.00006
    Test loss: 0.00006
    Test loss: 0.00005
    Test loss: 0.00005
    Test loss: 0.00005
    Test loss: 0.00005
    

Let's plot sample recostructed and actual usage data. To get reconstructed data we have to send our data throuhout both encoder and decoder. We can simply use `model(input_tensor_dataset)` for that. After that I have plot the reconstructed datasets with it's original shape.


```python
# Set the number of rows and columns
n_rows = 5
n_cols = 5

# Set the figure size and create the figure
plt.figure(figsize=(18, 8))

for i in range(n_rows):
    for j in range(n_cols):
        # Select a random index from the test data
        rand_id = np.random.choice(list(range(len(test_data))))

        # Get the original and sample data at the selected index
        original_data = test_data[rand_id].detach().numpy()
        constructed_data = model(test_data)[rand_id].detach().numpy()

        # Plot the original and sample data in the current subplot
        plt.subplot(n_rows, n_cols, i*n_cols + j + 1)
        plt.plot(constructed_data, color='red',alpha=0.4)
        plt.plot(original_data, color='blue',alpha=0.4)
        plt.grid(True)

# Show the plot
plt.tight_layout()
plt.show()
```


    
![png](README_files/README_21_0.png)
    


Alright, now we almost completed the denoicing part. Now we have clustering part left. Let's start with clustering.

These are the steps I follow,
 1. Fit the `KMeans` algorithms for original dataset and get the labels.
 2. Perform `PCA` on dataset and project dataset into 2d pane by using first 2 component of the PCA
 
This will repeat for the reconstructed data as well.


```python
from plotting.plotting_utils import plot_kmeans_pca,plot_time_series
```


```python
N_CLUSTERS = 5
```

Let's start with original dataset


```python
test_data_npy = test_data.detach().numpy()
labels = plot_kmeans_pca(test_data_npy,n_clusters=N_CLUSTERS)
```


    
![png](README_files/README_26_0.png)
    



    
![png](README_files/README_26_1.png)
    


Okay, now we have assigned each customers into segmentation based on their usage trend. Now it's time to plot sample dataset for each customer segment.


```python
clustered_df = pd.DataFrame(
    test_data_npy
)

clustered_df['cluster'] = labels
```


```python
plot_time_series(clustered_df)
```


    
![png](README_files/README_29_0.png)
    


After looking at above generated plot you may notice some reasonabl segmentation happend there. 

As an example, 
- Red segment users and green segment users have most inactive days. 
- Blue segement customers are relatively inactive in begining of the series and then they have gradually increased their usage over time.
- Orange and Purple users have relatively active pattern among other segmants. Among these two segments, the Orange group has the most active behavour compared to the Purple segment.

Let's repeat the same for reconstructed dataset as well.


```python
test_reconstructed = model(test_data).detach().numpy()
labels = plot_kmeans_pca(test_reconstructed,n_clusters=N_CLUSTERS)
```


    
![png](README_files/README_32_0.png)
    



    
![png](README_files/README_32_1.png)
    



```python
reconstructed_clustered_df = pd.DataFrame(
    test_data_npy
)

reconstructed_clustered_df['cluster'] = labels
```


```python
plot_time_series(reconstructed_clustered_df)
```


    
![png](README_files/README_34_0.png)
    

