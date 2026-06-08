#ifndef NETWORK_H_
#define NETWORK_H_

#include "layer.c"
#include "mnist.c"

#include <time.h>

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

  layer_relu(&network->layers_[SECOND_LAYER]);

  layer_forward_pass(&network->layers_[SECOND_LAYER],
                     &network->layers_[THIRD_LAYER]);

  layer_relu(&network->layers_[THIRD_LAYER]);

  layer_forward_pass(&network->layers_[THIRD_LAYER],
                     &network->layers_[FINAL_LAYER]);

  layer_softmax(&network->layers_[FINAL_LAYER]);
}

void network_backward(network_t *network, int expected) {
  layer_output_backwards(&network->layers_[FINAL_LAYER], expected);

  layer_compute_gradients(&network->layers_[THIRD_LAYER],
                          &network->layers_[FINAL_LAYER]);

  layer_compute_hidden(&network->layers_[THIRD_LAYER],
                       &network->layers_[FINAL_LAYER]);

  layer_compute_gradients(&network->layers_[SECOND_LAYER],
                          &network->layers_[THIRD_LAYER]);

  layer_compute_hidden(&network->layers_[SECOND_LAYER],
                       &network->layers_[THIRD_LAYER]);

  layer_compute_gradients(&network->layers_[INPUT_LAYER],
                          &network->layers_[SECOND_LAYER]);
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

#endif
