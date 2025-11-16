#include "mnist.cu"
#include "network.cu"

float mnist_test(network_t *network, const mnist_t *test) {
  int nr_correct = 0;
  for (int i = 0; i < test->testing_data_size_; ++i) {
    const mnist_input *in = &test->testing_data_[i];

    network_feed_image(network, &in->image_);

    network_forward(network);

    nr_correct += correct(&network->layers_[FINAL_LAYER], in->label_);

    fflush(stdout);
  }

  return (float)nr_correct / test->testing_data_size_;
}

__global__ void device_network_test(int total_num, int *number_correct,
                                    float *success) {
  *success = (*number_correct / (float)total_num) * 100.0f;
}

void mnist_device_test(network_t *network_d, const mnist_t *test,
                       float *success) {
  int *nr_correct = NULL;
  cudaMalloc((void **)&nr_correct, sizeof(int));

  for (int i = 0; i < test->testing_data_size_; ++i) {
    const mnist_input *in = &test->testing_data_[i];

    network_device_feed_image(network_d, &in->image_);

    network_device_forward(network_d);

    device_compute_correctness<<<1, 1>>>(network_d->layers_[FINAL_LAYER],
                                         in->label_, nr_correct);
  }

  device_network_test<<<1, 1>>>(test->testing_data_size_, nr_correct, success);

  cudaFree(nr_correct);
}

void final_test(network_t *network, const mnist_t *test) {
  int nr_correct = 0;

  printf("Running testing data...\n");

  for (int i = 0; i < test->testing_data_size_; ++i) {
    const mnist_input *in = &test->testing_data_[i];

    network_feed_image(network, &in->image_);

    network_forward(network);

    nr_correct += correct(&network->layers_[FINAL_LAYER], in->label_);

    printf("\rImages classified (%d/%d)", i + 1, test->testing_data_size_);

    fflush(stdout);
  }

  printf("\nModel got %d/%d correct!\n", nr_correct, test->testing_data_size_);
}

int main(void) {
  printf("MNIST Network\n");
  network_t network;
  hyper_parameters_t params{60, 0.1f, 10};
  network_init(&network, params);

  network_t network_device;
  network_device_init(&network_device, params);

  network_random_init(&network);

  network_copy_initial(&network_device, &network);

  mnist_t mnist_data;
  mnist_read_test_data("data/train-images.idx3-ubyte", &mnist_data);
  printf("Training data read\n");
  mnist_read_label_data("data/train-labels.idx1-ubyte", &mnist_data);
  printf("Training labels read\n");

  mnist_t test;
  mnist_read_test_data("data/t10k-images.idx3-ubyte", &test);
  printf("Testing data read\n");
  mnist_read_label_data("data/t10k-labels.idx1-ubyte", &test);
  printf("Testing labels read\n");

  int total_epochs = network.params_.epoch_count_;
  int batch_size = network.params_.batch_size_;

  int total_data = mnist_data.testing_data_size_;

  int groups = total_data / batch_size;

  float *accumulated_loss_d = NULL;
  cudaMalloc((void **)&accumulated_loss_d, sizeof(float));

  int *number_correct_d = NULL;
  cudaMalloc((void **)&number_correct_d, sizeof(int));

  float *accuracy_d = NULL;
  cudaMalloc((void **)&accuracy_d, sizeof(float));

  for (int i = 0; i < total_epochs; ++i) {

    mnist_shuffle(&mnist_data);
    printf("shuffle for epoch [%d] complete\n", i + 1);

    for (int j = 0; j < groups; ++j) {
      network_device_reset_gradients(&network_device);

      for (int k = 0; k < batch_size; ++k) {
        const mnist_input *in = &mnist_data.testing_data_[j * batch_size + k];
        network_device_feed_image(&network_device, &in->image_);

        int expected = in->label_;

        network_device_forward(&network_device);

        device_cross_entropy<<<1, 1>>>(
            accumulated_loss_d, network_device.layers_[FINAL_LAYER], expected);

        device_compute_correctness<<<1, 1>>>(
            network_device.layers_[FINAL_LAYER], expected, number_correct_d);

        network_device_backward(&network_device, expected);
      }

      network_device_update(&network_device);

      int number_correct = 0;

      cudaMemcpy(&number_correct, number_correct_d, sizeof(int),
                 cudaMemcpyDeviceToHost);

      printf("\r%d/%d completed, %d/%d correct", (j + 1) * batch_size,
             total_data, number_correct, (j + 1) * batch_size);

      fflush(stdout);
    }

    float accumulated_loss = 0.0f;

    cudaMemcpy(&accumulated_loss, accumulated_loss_d, sizeof(float),
               cudaMemcpyDeviceToHost);

    accumulated_loss /= total_data;

    mnist_device_test(&network_device, &test, accuracy_d);

    float accuracy = 0.0f;

    cudaMemcpy(&accuracy, accuracy_d, sizeof(float), cudaMemcpyDeviceToHost);

    printf("\nEpoch %d Completed, Loss: %.4f, Test Accuracy: %.4f%%\n", i + 1,
           accumulated_loss, accuracy);

    accumulated_loss = 0.0f;
    cudaMemcpy(accumulated_loss_d, &accumulated_loss, sizeof(float),
               cudaMemcpyHostToDevice);

    accuracy = 0.0f;
    cudaMemcpy(accuracy_d, &accuracy, sizeof(float), cudaMemcpyHostToDevice);

    int number_correct = 0;
    cudaMemcpy(number_correct_d, &number_correct, sizeof(int),
               cudaMemcpyHostToDevice);
  }

  cudaFree(accumulated_loss_d);
  cudaFree(number_correct_d);
  cudaFree(accuracy_d);

  device_layer_copy_to_host(network_device.layers_[SECOND_LAYER],
                            &network.layers_[SECOND_LAYER]);
  device_layer_copy_to_host(network_device.layers_[THIRD_LAYER],
                            &network.layers_[THIRD_LAYER]);
  device_layer_copy_to_host(network_device.layers_[FINAL_LAYER],
                            &network.layers_[FINAL_LAYER]);

  network_device_free(&network_device);

  final_test(&network, &test);

  mnist_free(&mnist_data);
  mnist_free(&test);

  printf("Done\n");
  return 0;
}
