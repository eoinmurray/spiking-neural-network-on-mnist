import os
import json
import numpy as np
import matplotlib.pyplot as plt
from src.load_data import load_mnist_data
from src.save_data import save_data
from src.process_encoding_data import process_encoding_data

def poisson_encode_speed(images, num_time_steps):
    images = images.numpy()
    images = images.reshape(images.shape[0], -1)
    spike_probabilities = np.repeat(images[:, None, :], num_time_steps, axis=1)
    return np.random.rand(*spike_probabilities.shape) < spike_probabilities

def lif_step(input_current, membrane_potential, membrane_decay, surrogate_grad_steepness, membrane_threshold = 1.0):
    membrane_potential = membrane_decay * membrane_potential + input_current
    spikes = membrane_potential > membrane_threshold

    exp_term = np.exp(-surrogate_grad_steepness * membrane_potential - 1.0)
    grad_surrogate = surrogate_grad_steepness * exp_term / (1.0 + exp_term) ** 2

    membrane_potential[spikes] = 0.0
    return spikes.astype(float), membrane_potential, grad_surrogate


def evaluate_model(data_loader, weights_input_to_hidden, weights_hidden_to_output, 
                  num_time_steps, num_hidden_neurons, num_output_neurons,
                  membrane_decay, surrogate_grad_steepness, membrane_threshold):
    correct_predictions = 0
    total_samples = 0
    last_record = {}

    for batch_idx, (images, labels) in enumerate(data_loader):
        is_last_batch = batch_idx == len(data_loader) - 1
        images = images.squeeze(1)

        encoded_spikes = poisson_encode_speed(images, num_time_steps).astype(float)

        membrane_potential_hidden = np.zeros((images.shape[0], num_hidden_neurons))
        membrane_potential_output = np.zeros((images.shape[0], num_output_neurons))
        output_spike_accumulator = np.zeros((images.shape[0], num_output_neurons))
        
        # For recording the last test image
        if is_last_batch:
            input_spike_trains = []
            hidden_spike_trains = []
            output_spike_trains = []
            hidden_membrane_potentials_before_reset = []
            output_membrane_potentials_before_reset = []
            hidden_membrane_potentials_after_reset = []
            output_membrane_potentials_after_reset = []
        
        for t in range(num_time_steps):
            if is_last_batch:
                last_idx = -1  # Index of the last image in the batch
                hidden_membrane_potentials_before_reset.append(membrane_potential_hidden[last_idx].tolist())
                output_membrane_potentials_before_reset.append(membrane_potential_output[last_idx].tolist())

            current_input_hidden = encoded_spikes[:, t, :] @ weights_input_to_hidden
            spikes_hidden, membrane_potential_hidden, _ = lif_step(current_input_hidden, membrane_potential_hidden, membrane_decay, surrogate_grad_steepness, membrane_threshold)
            current_input_output = spikes_hidden @ weights_hidden_to_output
            spikes_output, membrane_potential_output, _ = lif_step(current_input_output, membrane_potential_output, membrane_decay, surrogate_grad_steepness, membrane_threshold)
            output_spike_accumulator += spikes_output
            
            # Record data for the last image in the last batch
            if is_last_batch:
                last_idx = -1  # Index of the last image in the batch
                input_spike_trains.append(encoded_spikes[last_idx, t, :].tolist())
                hidden_spike_trains.append(spikes_hidden[last_idx].tolist())
                output_spike_trains.append(spikes_output[last_idx].tolist())
                hidden_membrane_potentials_after_reset.append(membrane_potential_hidden[last_idx].tolist())
                output_membrane_potentials_after_reset.append(membrane_potential_output[last_idx].tolist())

        predictions = np.argmax(output_spike_accumulator, axis=1)
        correct_predictions += (predictions == labels.numpy()).sum()
        total_samples += len(labels)
        
        # Save recorded data for the last image
        if is_last_batch:
            last_record = {
                "encoded_spikes": encoded_spikes[last_idx].tolist(),
                "input_spike_trains": np.array(input_spike_trains).T.tolist(),
                "hidden_spike_trains": np.array(hidden_spike_trains).T.tolist(),
                "output_spike_trains": np.array(output_spike_trains).T.tolist(),
                "hidden_membrane_potentials_before_reset": np.array(hidden_membrane_potentials_before_reset).T.tolist(),
                "output_membrane_potentials_before_reset": np.array(output_membrane_potentials_before_reset).T.tolist(),
                "hidden_membrane_potentials_after_reset": np.array(hidden_membrane_potentials_after_reset).T.tolist(),
                "output_membrane_potentials_after_reset": np.array(output_membrane_potentials_after_reset).T.tolist(),
                "true_label": int(labels.numpy()[last_idx]),
                "predicted_label": int(predictions[last_idx]),
                "image_data": images[last_idx].numpy().tolist(),
                "membrane_threshold": membrane_threshold
            }

    accuracy = 100 * correct_predictions / total_samples
    print(f"Test Accuracy: {accuracy:.2f}% for {total_samples} samples")    
    return accuracy, last_record

def main():
    FAST_MODE = False
    num_time_steps = 100
    num_hidden_neurons = 100
    num_output_neurons = 10
    membrane_decay = 0.9
    learning_rate = 1e-2
    batch_size = 128
    num_epochs = 5
    membrane_threshold=1.5
    surrogate_grad_steepness = 5.0
    dataset = 'MNIST' # 'MNIST' or 'FashionMNIST'

    if FAST_MODE:
        num_epochs = 10
        num_time_steps = 75
        batch_size = 10
        subset_size = 50
        print("FAST_MODE is ON. Training with reduced parameters.")
        train_loader, test_loader = load_mnist_data(batch_size, dataset, subset_size)
    else:
        print("FAST_MODE is OFF. Training with full parameters.")
        train_loader, test_loader = load_mnist_data(batch_size, dataset=dataset)

    num_input_neurons = next(iter(train_loader))[0][0][0].shape[0] ** 2

    weights_input_to_hidden = np.random.uniform(
        -np.sqrt(1 / num_input_neurons), np.sqrt(1 / num_input_neurons), size=(num_input_neurons, num_hidden_neurons)
    )
    weights_hidden_to_output = np.random.uniform(
        -np.sqrt(1 / num_hidden_neurons), np.sqrt(1 / num_hidden_neurons), size=(num_hidden_neurons, num_output_neurons)
    )

    weights_input_to_hidden_start = weights_input_to_hidden.copy()
    weights_hidden_to_output_start = weights_hidden_to_output.copy()

    losses = []
    accuracies = []
    grad_norms = {'input': [], 'output': []}

    accuracy, _ = evaluate_model(test_loader, weights_input_to_hidden, weights_hidden_to_output, num_time_steps, num_hidden_neurons, num_output_neurons, membrane_decay, surrogate_grad_steepness, membrane_threshold)
    print(f"Accuracy before training (with random weights): {accuracy:.2f}%")

    print(f"Training for {num_epochs} epochs with batch size {batch_size} and time steps {num_time_steps}.")
    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0

        for batch_iteration, (image_batch, label_batch) in enumerate(train_loader):
            is_first_epoch_and_batch = (epoch == 0 and batch_iteration == 0)
            is_last_epoch_and_batch = (epoch == num_epochs - 1 and batch_iteration == len(train_loader) - 1)

            if is_first_epoch_and_batch:
                print("First batch of the first epoch")

            if is_last_epoch_and_batch:
                print("Last batch of the last epoch")

            image_batch = image_batch.squeeze(1)
            encoded_spikes = poisson_encode_speed(image_batch, num_time_steps).astype(float)

            this_batch_size = image_batch.shape[0]

            membrane_potential_hidden = np.zeros((this_batch_size, num_hidden_neurons))
            membrane_potential_output = np.zeros((this_batch_size, num_output_neurons))
            output_spike_accumulator = np.zeros((this_batch_size, num_output_neurons))
            grad_weights_input = np.zeros_like(weights_input_to_hidden)
            grad_weights_output = np.zeros_like(weights_hidden_to_output)
            spike_history_hidden = []
            input_current_history_hidden = []         

            for t in range(num_time_steps):
                current_input_hidden = encoded_spikes[:, t, :] @ weights_input_to_hidden
                spikes_hidden, membrane_potential_hidden, grad_hidden = lif_step(current_input_hidden, membrane_potential_hidden, membrane_decay, surrogate_grad_steepness, membrane_threshold)
                spike_history_hidden.append(spikes_hidden)
                input_current_history_hidden.append(current_input_hidden)
                current_input_output = spikes_hidden @ weights_hidden_to_output
                spikes_output, membrane_potential_output, grad_output = lif_step(current_input_output, membrane_potential_output, membrane_decay, surrogate_grad_steepness, membrane_threshold)
                output_spike_accumulator += spikes_output

            softmax_numerators = np.exp(output_spike_accumulator - np.max(output_spike_accumulator, axis=1, keepdims=True))
            probabilities = softmax_numerators / softmax_numerators.sum(axis=1, keepdims=True)
            one_hot_targets = np.zeros_like(probabilities)
            one_hot_targets[np.arange(len(label_batch)), label_batch.numpy()] = 1
            loss = -np.mean(np.sum(one_hot_targets * np.log(probabilities + 1e-9), axis=1))
            grad_logits = (probabilities - one_hot_targets) / batch_size

            for t in reversed(range(num_time_steps)):
                spikes_hidden = spike_history_hidden[t]
                input_current_hidden = input_current_history_hidden[t]
                current_input_output = spikes_hidden @ weights_hidden_to_output

                _, _, grad_output = lif_step(current_input_output, np.zeros_like(membrane_potential_output), membrane_decay, surrogate_grad_steepness, membrane_threshold)
                grad_output_current = grad_logits * grad_output
                grad_weights_output += spikes_hidden.T @ grad_output_current
                grad_spikes_hidden = grad_output_current @ weights_hidden_to_output.T

                _, _, grad_hidden = lif_step(input_current_hidden, np.zeros_like(membrane_potential_hidden), membrane_decay, surrogate_grad_steepness, membrane_threshold)
                grad_input_hidden = grad_spikes_hidden * grad_hidden
                grad_weights_input += encoded_spikes[:, t, :].T @ grad_input_hidden

            if batch_iteration % 50 == 0:
                grad_norms['input'].append(np.linalg.norm(grad_weights_input))
                grad_norms['output'].append(np.linalg.norm(grad_weights_output))

            weights_input_to_hidden -= learning_rate * grad_weights_input
            weights_hidden_to_output -= learning_rate * grad_weights_output

            epoch_loss += loss
            num_batches += 1            

        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)

        accuracy, _ = evaluate_model(test_loader, weights_input_to_hidden, weights_hidden_to_output, num_time_steps, num_hidden_neurons, num_output_neurons, membrane_decay, surrogate_grad_steepness, membrane_threshold)

        accuracies.append(accuracy)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    accuracy, last_image_record = evaluate_model(test_loader, weights_input_to_hidden, weights_hidden_to_output, 
                                 num_time_steps, num_hidden_neurons, num_output_neurons, 
                                 membrane_decay, surrogate_grad_steepness, membrane_threshold)
    
    weights_input_to_hidden_end = weights_input_to_hidden.copy()
    weights_hidden_to_output_end = weights_hidden_to_output.copy()

    encoding_data = process_encoding_data(test_loader, poisson_encode_speed, num_time_steps, output_file=os.path.join("reports", "spike_encoding_data.json"))

    save_object = {
        "weights_input_to_hidden_start": weights_input_to_hidden_start.tolist(),
        "weights_hidden_to_output_start": weights_hidden_to_output_start.tolist(),
        "weights_input_to_hidden_end": weights_input_to_hidden_end.tolist(),
        "weights_hidden_to_output_end": weights_hidden_to_output_end.tolist(),
        "grad_norms": grad_norms,
        "losses": losses,
        "accuracies": accuracies,
        "time_steps": list(range(num_time_steps)),
        "last_test_image_record": last_image_record,
        "encoding_data": encoding_data
    }

    save_data("reports", save_object)
    print(f"Final Test Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()
