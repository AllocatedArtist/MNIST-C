#include "mnist.c"
#include "network.c"

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
  network_init(&network, (hyper_parameters_t){.batch_size_ = 60,
                                              .learning_rate_ = 0.1f,
                                              .epoch_count_ = 5});

  network_random_init(&network);

  mnist_t mnist_data;
  mnist_read_test_data("../train-images-idx3-ubyte/train-images.idx3-ubyte",
                       &mnist_data);
  printf("Training data read\n");
  mnist_read_label_data("../train-labels-idx1-ubyte/train-labels.idx1-ubyte",
                        &mnist_data);
  printf("Training labels read\n");

  mnist_t test;
  mnist_read_test_data("../t10k-images-idx3-ubyte/t10k-images.idx3-ubyte",
                       &test);
  printf("Testing data read\n");
  mnist_read_label_data("../t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte",
                        &test);
  printf("Testing labels read\n");

  int total_epochs = network.params_.epoch_count_;
  int batch_size = network.params_.batch_size_;

  int total_data = mnist_data.testing_data_size_;

  int groups = total_data / batch_size;

  for (int i = 0; i < total_epochs; ++i) {
    int number_correct = 0;

    float accumulated_loss = 0.0f;

    mnist_shuffle(&mnist_data);
    printf("shuffle for epoch [%d] complete\n", i + 1);

    for (int j = 0; j < groups; ++j) {
      network_reset_gradients(&network);

      for (int k = 0; k < batch_size; ++k) {
        const mnist_input *in = &mnist_data.testing_data_[j * batch_size + k];
        network_feed_image(&network, &in->image_);

        int expected = in->label_;

        network_forward(&network);

        accumulated_loss +=
            cross_entropy(&network.layers_[FINAL_LAYER], expected);
        number_correct += correct(&network.layers_[FINAL_LAYER], expected);

        network_backward(&network, expected);
      }

      network_update(&network);

      printf("\r%d/%d completed, %d/%d correct", (j + 1) * batch_size,
             total_data, number_correct, (j + 1) * batch_size);
      fflush(stdout);
    }

    accumulated_loss /= total_data;

    float accuracy = mnist_test(&network, &test) * 100.0f;

    printf("\nEpoch %d Completed, Loss: %.4f, Test Accuracy: %.4f%%\n", i + 1,
           accumulated_loss, accuracy);
  }

  final_test(&network, &test);

  mnist_free(&mnist_data);
  mnist_free(&test);

  printf("Done\n");
  return 0;
}
