print("Training the SNN...")
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
        
        membrane_potential_hidden = np.zeros((image_batch.shape[0], num_hidden_neurons))
        membrane_potential_output = np.zeros((image_batch.shape[0], num_output_neurons))
        output_spike_accumulator = np.zeros((image_batch.shape[0], num_output_neurons))
        grad_weights_input = np.zeros_like(weights_input_to_hidden)
        grad_weights_output = np.zeros_like(weights_hidden_to_output)
        spike_history_hidden = []
        input_current_history_hidden = []            

        for t in range(num_time_steps):
            current_input_hidden = encoded_spikes[:, t, :] @ weights_input_to_hidden
            spikes_hidden, membrane_potential_hidden, grad_hidden = lif_step(current_input_hidden, membrane_potential_hidden, membrane_decay, surrogate_grad_steepness)
            spike_history_hidden.append(spikes_hidden)
            input_current_history_hidden.append(current_input_hidden)
            current_input_output = spikes_hidden @ weights_hidden_to_output
            spikes_output, membrane_potential_output, grad_output = lif_step(current_input_output, membrane_potential_output, membrane_decay, surrogate_grad_steepness)
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
            
            _, _, grad_output = lif_step(current_input_output, np.zeros_like(membrane_potential_output), membrane_decay, surrogate_grad_steepness)
            grad_output_current = grad_logits * grad_output
            grad_weights_output += spikes_hidden.T @ grad_output_current
            grad_spikes_hidden = grad_output_current @ weights_hidden_to_output.T

            _, _, grad_hidden = lif_step(input_current_hidden, np.zeros_like(membrane_potential_hidden), membrane_decay, surrogate_grad_steepness)
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

    accuracy, _, _, _ = evaluate_model(test_loader, weights_input_to_hidden, weights_hidden_to_output, num_time_steps, num_hidden_neurons, num_output_neurons, membrane_decay, surrogate_grad_steepness)
    accuracies.append(accuracy)

    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")