import numpy as np
import json

def process_encoding_data(test_dataset, poisson_encode, num_time_steps, output_file='spike_encoding_data.json'):
    """
    Encodes the first 3 examples from test_dataset using Poisson encoding,
    collects spike-train data, and saves it to a JSON file.

    Returns the path to the JSON file created.
    """
    results = []
    num_examples = 3

    # Get a data iterator from the DataLoader
    test_iterator = iter(test_dataset)
    
    for i in range(num_examples):
        img, label = next(test_iterator)
        # Process only the first image in the batch
        img = img[0].squeeze(0)
        label = label[0]
        # Generate Poisson-encoded spikes: shape [num_time_steps, num_neurons]
        encoded_spikes = poisson_encode(img.unsqueeze(0), num_time_steps)[0]

        spike_times = []
        neuron_ids = []
        num_neurons = encoded_spikes.shape[1]

        # Collect spike times and neuron indices
        for neuron in range(num_neurons):
            times = np.where(encoded_spikes[:, neuron] == 1)[0]
            if times.size > 0:
                spike_times.extend(times.tolist())
                neuron_ids.extend([neuron] * len(times))

        # Append example data, converting arrays to lists for JSON serialization
        results.append({
            'label': int(label),
            'image': img.numpy().tolist(),
            'spike_times': spike_times,
            'neuron_ids': neuron_ids,
            'encoded_spikes': encoded_spikes.tolist()
        })

    return results