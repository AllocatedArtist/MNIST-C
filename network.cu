#ifndef NETWORK_H_
#define NETWORK_H_

#include "layer.cu"
#include "mnist.cu"

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

void network_copy_initial(network_t *network_d, network_t *network_h) {
  device_layer_copy_from_host(network_d->layers_[INPUT_LAYER],
                              &network_h->layers_[INPUT_LAYER]);

  device_layer_copy_from_host(network_d->layers_[SECOND_LAYER],
                              &network_h->layers_[SECOND_LAYER]);

  device_layer_copy_from_host(network_d->layers_[THIRD_LAYER],
                              &network_h->layers_[THIRD_LAYER]);

  device_layer_copy_from_host(network_d->layers_[FINAL_LAYER],
                              &network_h->layers_[FINAL_LAYER]);
}

void network_device_init(network_t *network, hyper_parameters_t params) {
  network->params_ = params;

  device_layer_init(&network->layers_[INPUT_LAYER], 0, PIXEL_COUNT);
  device_layer_init(&network->layers_[SECOND_LAYER], PIXEL_COUNT, SL_SIZE);
  device_layer_init(&network->layers_[THIRD_LAYER], SL_SIZE, TL_SIZE);
  device_layer_init(&network->layers_[FINAL_LAYER], TL_SIZE, DIGIT_COUNT);
}

void network_device_free(network_t *network) {
  device_layer_free(network->layers_[INPUT_LAYER]);
  device_layer_free(network->layers_[SECOND_LAYER]);
  device_layer_free(network->layers_[THIRD_LAYER]);
  device_layer_free(network->layers_[FINAL_LAYER]);
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

  layer_softmax(&network->layers_[FINAL_LAYER]);
}

void network_device_forward(network_t *network) {

  device_layer_forward_pass<<<1, SL_SIZE>>>(network->layers_[INPUT_LAYER],
                                            network->layers_[SECOND_LAYER]);

  device_layer_sigmoid<<<1, SL_SIZE>>>(network->layers_[SECOND_LAYER]);

  device_layer_forward_pass<<<1, TL_SIZE>>>(network->layers_[SECOND_LAYER],
                                            network->layers_[THIRD_LAYER]);

  device_layer_sigmoid<<<1, TL_SIZE>>>(network->layers_[THIRD_LAYER]);

  device_layer_forward_pass<<<1, DIGIT_COUNT>>>(network->layers_[THIRD_LAYER],
                                                network->layers_[FINAL_LAYER]);

  device_softmax<<<1, DIGIT_COUNT>>>(network->layers_[FINAL_LAYER]);
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

void network_device_backward(network_t *network, int expected) {
  device_output_backwards<<<1, DIGIT_COUNT>>>(network->layers_[FINAL_LAYER],
                                              expected);

  device_compute_gradients<<<1, DIGIT_COUNT>>>(network->layers_[THIRD_LAYER],
                                               network->layers_[FINAL_LAYER]);

  device_compute_hidden<<<1, TL_SIZE>>>(network->layers_[THIRD_LAYER],
                                        network->layers_[FINAL_LAYER]);

  device_compute_gradients<<<1, TL_SIZE>>>(network->layers_[SECOND_LAYER],
                                           network->layers_[THIRD_LAYER]);

  device_compute_hidden<<<1, SL_SIZE>>>(network->layers_[SECOND_LAYER],
                                        network->layers_[THIRD_LAYER]);

  device_compute_gradients<<<1, SL_SIZE>>>(network->layers_[INPUT_LAYER],
                                           network->layers_[SECOND_LAYER]);
}

__global__ void device_reset_gradient(layer_t layer) {
  for (int j = 0; j < layer.input_count_ * layer.neuron_count_; ++j) {
    layer.weight_gradients_[j] = 0.0f;
  }
  for (int j = 0; j < layer.neuron_count_; ++j) {
    layer.bias_gradients_[j] = 0.0f;
  }
}

void network_device_reset_gradients(network_t *network) {
  for (int i = 1; i < LAYER_COUNT; ++i) {
    device_reset_gradient<<<1, 1>>>(network->layers_[i]);
  }
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
      assert(!isinf(layer->weights_[j]));
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

void network_device_update(network_t *network) {
  float batch_size = network->params_.batch_size_;
  float learning_rate = network->params_.learning_rate_;

  device_layer_update<<<1, DIGIT_COUNT>>>(network->layers_[FINAL_LAYER],
                                          batch_size, learning_rate);
  device_layer_update<<<1, TL_SIZE>>>(network->layers_[THIRD_LAYER], batch_size,
                                      learning_rate);
  device_layer_update<<<1, SL_SIZE>>>(network->layers_[SECOND_LAYER],
                                      batch_size, learning_rate);
}

__global__ void device_feed_image(layer_t input, uint8_t *data_d) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < input.neuron_count_) {
    input.activations_[idx] = (float)data_d[idx] / 255.0f;
  }
}

void network_device_feed_image(network_t *network, const image_t *image) {
  assert(image->data_size_ > 0 && "invalid image");
  uint8_t *data_d = NULL;
  cudaMalloc((void **)&data_d, sizeof(uint8_t) * image->data_size_);
  cudaMemcpy(data_d, image->data_, sizeof(uint8_t) * image->data_size_,
             cudaMemcpyHostToDevice);
  device_feed_image<<<1, image->data_size_>>>(network->layers_[INPUT_LAYER],
                                              data_d);
  cudaFree(data_d);
}

void network_feed_image(network_t *network, const image_t *image) {
  layer_t *input_layer = &network->layers_[0];
  assert(input_layer->neuron_count_ == image->data_size_);

  for (int i = 0; i < image->data_size_; ++i) {
    input_layer->activations_[i] = (float)image->data_[i] / 255.0f;
  }
}

__global__ void device_compute_correctness(layer_t output, int expected,
                                           int *correct) {
  int max_index = 0;
  float max = output.activations_[0];
  for (int i = 1; i < output.neuron_count_; ++i) {
    if (output.activations_[i] > max) {
      max = output.activations_[i];
      max_index = i;
    }
  }

  *correct += (max_index == expected);
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
