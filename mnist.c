#include <assert.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

#include <stdint.h>

#include <string.h>
#include <time.h>

#define _USE_MATH_DEFINES
#include <math.h>

#define LAYER_COUNT 4

#define PIXEL_COUNT 28 * 28
#define SL_SIZE 300
#define TL_SIZE 100
#define DIGIT_COUNT 10

#define INPUT_LAYER 0
#define SECOND_LAYER 1
#define THIRD_LAYER 2
#define FINAL_LAYER 3

typedef struct {
  uint8_t *data_;
  int data_size_;
} image_t;

typedef struct {
  image_t image_;
  int label_;
} mnist_input;

typedef struct {
  mnist_input *testing_data_;
  int testing_data_size_;
} mnist_t;

typedef enum {
  UBYTE = 0x08,
  SBYTE = 0x09,
  SHORT = 0x0B,
  INT = 0x0C,
  FLOAT = 0x0D,
  DOUBLE = 0x0E
} mnist_magic_number;

void mnist_read_test_data(const char *path, mnist_t *mnist) {
  FILE *fp = fopen(path, "rb");
  assert(fp != NULL && "File not found");

  uint32_t magic_number;

  fread(&magic_number, sizeof(uint32_t), 1, fp);
  magic_number = __builtin_bswap32(magic_number);

  uint32_t type = (magic_number >> 8) & 0xFF;

  assert(type == UBYTE && "Expected unsigned byte");

  uint32_t dimensions_count = magic_number & 0xFF;

  uint32_t size = 1;
  for (int i = 0; i < dimensions_count; ++i) {
    uint32_t dimension_size = 0;
    fread(&dimension_size, sizeof(uint32_t), 1, fp);
    dimension_size = __builtin_bswap32(dimension_size);

    size *= dimension_size;
  }

  uint8_t *all_bytes = malloc(size);
  assert(all_bytes != NULL && "Unable to alloc memory");

  uint32_t bytes_read = fread(all_bytes, sizeof(uint8_t), size, fp);

  assert(bytes_read == size && "Unable to read all data");

  int img_count = size / (PIXEL_COUNT);

  mnist->testing_data_size_ = img_count;
  mnist->testing_data_ = malloc(sizeof(mnist_input) * img_count);

  for (int i = 0; i < img_count; ++i) {
    int offset = PIXEL_COUNT;
    image_t *current = &mnist->testing_data_[i].image_;

    current->data_ = malloc(PIXEL_COUNT);
    memcpy(current->data_, all_bytes + i * PIXEL_COUNT, PIXEL_COUNT);
    current->data_size_ = PIXEL_COUNT;
  }

  free(all_bytes);

  fclose(fp);
}

void mnist_read_label_data(const char *path, mnist_t *mnist) {
  FILE *fp = fopen(path, "rb");
  assert(fp != NULL && "File not found");

  uint32_t magic_number;

  fread(&magic_number, sizeof(uint32_t), 1, fp);
  magic_number = __builtin_bswap32(magic_number);

  uint32_t type = (magic_number >> 8) & 0xFF;

  assert(type == UBYTE && "Expected unsigned byte");

  uint32_t dimensions_count = magic_number & 0xFF;

  uint32_t size = 1;
  for (int i = 0; i < dimensions_count; ++i) {
    uint32_t dimension_size = 0;
    fread(&dimension_size, sizeof(uint32_t), 1, fp);
    dimension_size = __builtin_bswap32(dimension_size);

    size *= dimension_size;
  }

  uint8_t *all_bytes = malloc(size);

  fread(all_bytes, sizeof(uint8_t), size, fp);

  for (int i = 0; i < size; ++i) {
    mnist->testing_data_[i].label_ = all_bytes[i];
  }

  free(all_bytes);

  fclose(fp);
}

void mnist_free(mnist_t *mnist) {
  for (int i = 0; i < mnist->testing_data_size_; ++i) {
    free(mnist->testing_data_[i].image_.data_);
  }
  free(mnist->testing_data_);
}

typedef struct {
  float *activations_;

  float *biases_;
  float *weights_;

  float *inputs_;

  float *dl_;

  float *weight_gradients_;
  float *bias_gradients_;

  int neuron_count_;
  int input_count_;
} layer_t;

void layer_init(layer_t *layer, int input_size, int neuron_size) {
  layer->inputs_ = malloc(sizeof(float) * neuron_size);
  layer->activations_ = malloc(sizeof(float) * neuron_size);

  layer->weights_ = NULL;
  layer->weight_gradients_ = NULL;
  layer->biases_ = NULL;
  layer->bias_gradients_ = NULL;
  layer->dl_ = NULL;

  if (input_size > 0) {
    layer->weights_ = malloc(sizeof(float) * input_size * neuron_size);
    layer->weight_gradients_ = malloc(sizeof(float) * input_size * neuron_size);

    layer->biases_ = malloc(sizeof(float) * neuron_size);
    layer->bias_gradients_ = malloc(sizeof(float) * neuron_size);

    layer->dl_ = malloc(sizeof(float) * neuron_size);
  }

  layer->neuron_count_ = neuron_size;
  layer->input_count_ = input_size;
}

void layer_forward_pass(const layer_t *input, layer_t *output) {
  int input_size = input->neuron_count_;
  int output_size = output->neuron_count_;

  for (int y = 0; y < output_size; ++y) {
    float dp = output->biases_[y];

    for (int x = 0; x < input_size; ++x) {
      float act = input->activations_[x];
      float weight = output->weights_[y * input_size + x];

      dp += act * weight;
    }

    if (fabs(dp) == INFINITY) {
      printf("Input: %d neurons, Output: %d neurons\n", input->neuron_count_,
             output->neuron_count_);
      printf("Input:\n");
      for (int i = 0; i < input->neuron_count_; ++i) {
        printf("%f\n", input->activations_[i]);
      }
      printf("Weights:\n");
      for (int i = 0; i < output->neuron_count_ * input->neuron_count_; ++i) {
        printf("%f\n", output->weights_[i]);
      }
      printf("Biases:\n");
      for (int i = 0; i < output->neuron_count_; ++i) {
        printf("%f\n", output->biases_[i]);
      }
    }

    assert(fabs(dp) != INFINITY && "Invalid dot product");

    output->inputs_[y] = dp;
  }
}

void layer_relu(layer_t *output) {
  int output_size = output->neuron_count_;
  for (int i = 0; i < output_size; ++i) {
    float input = output->inputs_[i];
    output->activations_[i] = input > 0.0f ? input : 0.0f;
  }
}

void layer_sigmoid(layer_t *output) {
  int output_size = output->neuron_count_;
  for (int i = 0; i < output_size; ++i) {
    float input = output->inputs_[i];
    output->activations_[i] = 1.0f / (1 + expf(-input));
  }
}

void layer_softmax(layer_t *output) {
  float *inputs = output->inputs_;
  int input_size = output->neuron_count_;
  float max_component = output->inputs_[0];
  for (int i = 1; i < input_size; ++i) {
    float input = inputs[i];
    max_component = fmaxf(input, max_component);
  }

  float divisor = 0.0f;
  for (int i = 0; i < input_size; ++i) {
    float input = inputs[i];
    divisor += expf(input - max_component);
  }

  // TODO: Address small divisor
  divisor = 1 / fmaxf(1e-8f, divisor);

  assert(divisor > 0.0f && "Divide by zero error");

  for (int i = 0; i < input_size; ++i) {
    float numerator = exp(inputs[i] - max_component);
    output->activations_[i] = numerator * divisor;
  }
}

float layer_cross_entropy(layer_t *output, int expected) {
  assert(output->neuron_count_ == DIGIT_COUNT);

  float loss = 0.0f;

  int expected_val[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  expected_val[expected] = 1;

  for (int i = 0; i < DIGIT_COUNT; ++i) {
    loss -= output->activations_[i] * log(fmaxf(1e-6, expected_val[i]));
  }

  return loss;
}

float layer_mean_squared_error(layer_t *output, int expected) {
  assert(output->neuron_count_ == DIGIT_COUNT);

  float loss = 0.0f;

  int expected_val[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  expected_val[expected] = 1;

  const float cf = 1.0f / DIGIT_COUNT;

  float sum = 0.0f;
  for (int i = 0; i < DIGIT_COUNT; ++i) {
    sum += powf((output->activations_[i] - expected), 2.0f);
  }

  return cf * sum;
}

void layer_cross_entropy_backwards(layer_t *output, int expected) {
  assert(expected >= 0 && expected < 10 && "Invalid expectation.");
  int expected_val[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  expected_val[expected] = 1.0f;

  for (int i = 0; i < output->neuron_count_; ++i) {
    output->dl_[i] = output->activations_[i] - expected_val[i];
  }
}

// Use with sigmoid sequence
void layer_update_backwards(layer_t *current, layer_t *next,
                            float learning_rate) {
  float *dl_array = current->dl_;
  float *next_dl_array = next->dl_;

  int current_layer_size = current->neuron_count_;
  int next_layer_size = next->neuron_count_;

  // update weights + biases of next layer

  for (int y = 0; y < next_layer_size; ++y) {
    float dl = next_dl_array[y];
    for (int x = 0; x < current_layer_size; ++x) {
      float current_act = current->activations_[x];
      next->weights_[x + y * current_layer_size] -=
          learning_rate * dl * current_act;
    }
    next->biases_[y] -= learning_rate * dl;
  }

  if (current->dl_ != NULL) {
    for (int x = 0; x < current->neuron_count_; ++x) {
      float delta_h = 0.0f;
      for (int y = 0; y < next->neuron_count_; ++y) {
        delta_h +=
            next->weights_[y * current->neuron_count_ + x] * next->dl_[y];
      }

      float act = current->activations_[x];
      float sigmoid_derivative = act * (1 - act);
      current->dl_[x] = delta_h * sigmoid_derivative;
    }
  }
}

void layer_relu_backwards(layer_t *current, layer_t *next) {
  float *dl_array = current->dl_;

  int current_layer_size = current->neuron_count_;
  int next_layer_size = next->neuron_count_;

  for (int y = 0; y < current_layer_size; ++y) {
    float dl = 0.0f;
    for (int x = 0; x < next_layer_size; ++x) {
      dl += next->weights_[y + x * current_layer_size] * next->dl_[x];
    }

    current->dl_[y] = current->inputs_[y] > 0 ? dl : 0;
  }
}

void layer_compute_gradients(layer_t *current, layer_t *previous) {
  float *dl_array = current->dl_;
  float current_layer_size = current->neuron_count_;
  for (int y = 0; y < current_layer_size; ++y) {
    float dl = current->dl_[y];
    current->bias_gradients_[y] += dl;
    for (int x = 0; x < previous->neuron_count_; ++x) {
      current->weight_gradients_[x + y * current->neuron_count_] +=
          dl * previous->activations_[x];
    }
  }
}

void layer_update(layer_t *current, float batch_size, float learning_rate) {

  float cf = learning_rate / batch_size;

  int input_count = current->input_count_;

  for (int i = 0; i < current->neuron_count_; ++i) {
    current->biases_[i] -= cf * current->bias_gradients_[i];
    for (int j = 0; j < input_count; ++j) {
      int current_index = j + i * current->input_count_;
      current->weights_[current_index] -=
          cf * current->weight_gradients_[current_index];
    }
  }
}

typedef struct {
  float batch_size_;
  float learning_rate_;
  float epoch_count_;
} hyper_parameters_t;

typedef struct {
  layer_t layers_[LAYER_COUNT];
  hyper_parameters_t params_;
} network_t;

void network_init(network_t *network, hyper_parameters_t params) {
  network->params_ = params;

  layer_init(&network->layers_[INPUT_LAYER], 0, PIXEL_COUNT);
  layer_init(&network->layers_[SECOND_LAYER], PIXEL_COUNT, SL_SIZE);
  layer_init(&network->layers_[THIRD_LAYER], SL_SIZE, TL_SIZE);
  layer_init(&network->layers_[FINAL_LAYER], TL_SIZE, DIGIT_COUNT);
}

void layer_check(const layer_t *layer, int index) {
  for (int i = 0; i < layer->neuron_count_; ++i) {
    if (isinf(layer->biases_[i]) || isnan(layer->biases_[i])) {
      fprintf(stderr, "Error in layer: %d (Invalid bias)\n", index);
    }
    if (isinf(layer->activations_[i]) || isnan(layer->activations_[i])) {
      fprintf(stderr, "Error in layer: %d (Invalid activation)\n", index);
    }
  }

  for (int i = 0; i < layer->neuron_count_ * layer->input_count_; ++i) {
    if (isinf(layer->weights_[i]) || isnan(layer->weights_[i])) {
      fprintf(stderr, "Error in layer: %d (Invalid weights)\n", index);
    }
  }
}

void network_forward(network_t *network) {

  layer_forward_pass(&network->layers_[INPUT_LAYER],
                     &network->layers_[SECOND_LAYER]);

  layer_sigmoid(&network->layers_[SECOND_LAYER]);

  layer_forward_pass(&network->layers_[SECOND_LAYER],
                     &network->layers_[THIRD_LAYER]);

  layer_sigmoid(&network->layers_[THIRD_LAYER]);

  layer_forward_pass(&network->layers_[THIRD_LAYER],
                     &network->layers_[FINAL_LAYER]);

  layer_sigmoid(&network->layers_[FINAL_LAYER]);

  // layer_softmax(&network->layers_[FINAL_LAYER]);
}

void network_backward(network_t *network, int expected) {
  layer_cross_entropy_backwards(&network->layers_[FINAL_LAYER], expected);
  layer_compute_gradients(&network->layers_[FINAL_LAYER],
                          &network->layers_[THIRD_LAYER]);

  layer_relu_backwards(&network->layers_[THIRD_LAYER],
                       &network->layers_[FINAL_LAYER]);
  layer_compute_gradients(&network->layers_[THIRD_LAYER],
                          &network->layers_[SECOND_LAYER]);

  layer_relu_backwards(&network->layers_[SECOND_LAYER],
                       &network->layers_[THIRD_LAYER]);
  layer_compute_gradients(&network->layers_[SECOND_LAYER],
                          &network->layers_[INPUT_LAYER]);
}

void network_reset_gradients(network_t *network) {
  for (int i = 1; i < LAYER_COUNT; ++i) {
    layer_t *layer = &network->layers_[i];
    assert(layer->bias_gradients_ != NULL && layer->weight_gradients_ != NULL);
    for (int j = 0; j < layer->input_count_ * layer->neuron_count_; ++j) {
      layer->weight_gradients_[j] = 0.0f;
    }
    for (int j = 0; j < layer->neuron_count_; ++j) {
      layer->bias_gradients_[j] = 0.0f;
    }
  }
}

float randn() {
  float random = ((float)rand() / RAND_MAX) - 0.5;
  return random;
}

void network_random_init(network_t *network) {
  srand(time(NULL));

  for (int i = 1; i < LAYER_COUNT; ++i) {
    layer_t *layer = &network->layers_[i];
    assert(layer->biases_ != NULL && layer->weights_ != NULL);
    for (int j = 0; j < layer->input_count_ * layer->neuron_count_; ++j) {
      layer->weights_[j] = randn();
      assert(layer->weights_[j] != INFINITY);
      assert(layer->weights_[j] >= -0.5 && layer->weights_[j] <= 0.5);
    }

    for (int j = 0; j < layer->neuron_count_; ++j) {
      layer->biases_[j] = 0.0f;
      // assert(layer->biases_[j] >= -0.5 && layer->biases_[j] <= 0.5);
    }
  }
}

void network_update(network_t *network) {
  float batch_size = network->params_.batch_size_;
  float learning_rate = network->params_.learning_rate_;

  layer_update(&network->layers_[FINAL_LAYER], batch_size, learning_rate);
  layer_update(&network->layers_[THIRD_LAYER], batch_size, learning_rate);
  layer_update(&network->layers_[SECOND_LAYER], batch_size, learning_rate);
}

void network_feed_image(network_t *network, const image_t *image) {
  layer_t *input_layer = &network->layers_[0];
  assert(input_layer->neuron_count_ == image->data_size_);

  for (int i = 0; i < image->data_size_; ++i) {
    input_layer->activations_[i] = (float)image->data_[i] / 255.0f;
  }
}

int correct(const layer_t *output, int expected) {
  int max_index = 0;
  float max = output->activations_[0];
  for (int i = 1; i < output->neuron_count_; ++i) {
    if (output->activations_[i] > max) {
      max = output->activations_[i];
      max_index = i;
    }
  }

  return max_index == expected;
}

int main(void) {
  printf("MNIST Network\n");
  network_t network;
  network_init(&network, (hyper_parameters_t){.batch_size_ = 60,
                                              .learning_rate_ = 0.01f,
                                              .epoch_count_ = 3});

  network_random_init(&network);

  mnist_t mnist_data;
  mnist_read_test_data("../train-images-idx3-ubyte/train-images.idx3-ubyte",
                       &mnist_data);
  printf("Test data read\n");
  mnist_read_label_data("../train-labels-idx1-ubyte/train-labels.idx1-ubyte",
                        &mnist_data);
  printf("Test label data read\n");

  int total_epochs = network.params_.epoch_count_;
  int batch_size = network.params_.batch_size_;
  int total_data = mnist_data.testing_data_size_;

  int groups = total_data / batch_size;

  for (int i = 0; i < total_epochs; ++i) {
    float accumulated_loss = 0.0f;
    int number_correct = 0;
    for (int j = 0; j < groups; ++j) {
      network_reset_gradients(&network);

      float batch_loss = 0.0f;
      for (int k = 0; k < batch_size; ++k) {
        const mnist_input *current_input =
            &mnist_data.testing_data_[j * batch_size + k];

        network_feed_image(&network, &current_input->image_);

        network_forward(&network);

        batch_loss += layer_cross_entropy(&network.layers_[FINAL_LAYER],
                                          current_input->label_);
        number_correct +=
            correct(&network.layers_[FINAL_LAYER], current_input->label_);

        network_backward(&network, current_input->label_);
      }
      accumulated_loss += batch_loss / batch_size;
      network_update(&network);

      int num_batches = (j + 1) * batch_size;
      float progress = (float)num_batches / total_data;

      int bar_width = 50;

      int filled_length = (int)(bar_width * progress);

      printf("\r[");
      for (int i = 0; i < bar_width; ++i) {
        if (i < filled_length) {
          printf("#");
        } else {
          printf(".");
        }
      }
      printf("] Epoch %d, Accuracy: %d/%d", i, number_correct, total_data);
      fflush(stdout);
    }
    printf("\nEpoch %d/%d (Completed), Loss: %.4f\n, Accuracy: %.2f", i + 1,
           total_epochs, accumulated_loss, number_correct / (float)total_data);
  }

  mnist_free(&mnist_data);

  printf("Done\n");
  return 0;
}
