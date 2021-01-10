#ifndef UDNN_LAYER_HH
#define UDNN_LAYER_HH

#include "tensor.hh"
#include <cmath>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <algorithm>
#include <limits>

enum class DType { Int8, Int16, Int32, Int64, Float, Double };

class LayerBase {
public:
  virtual TensorSize in_size() const = 0;
  virtual TensorSize out_size() const = 0;
  virtual DType in_type() const = 0;
  virtual DType out_type() const = 0;

  virtual const TensorBase *out_base() const = 0;

  std::string name;

  inline LayerBase *connect(LayerBase *next) {
    if (!next) {
      next_ = nullptr;
      return nullptr;
    }
    if (next->in_size() != out_size())
      throw std::invalid_argument(
          "Tensor dimension mismatch: " + next->in_size().str() + " -> " +
          out_size().str());
    if (next->in_type() != out_type())
      throw std::invalid_argument("Tensor type mismatch");
    next_ = next;
    return next;
  }
  inline LayerBase *next() const { return next_; }
  virtual void forward(void *data) = 0;

  ~LayerBase() = default;

private:
  LayerBase *next_ = nullptr;
};

template <typename T> class Layer : public LayerBase {
public:
  inline Layer(const TensorSize &in_size, const TensorSize &out_size)
      : in_size_(in_size), out_(out_size) {}

  inline Layer() = default;

  inline TensorSize in_size() const override { return in_size_; }
  inline TensorSize out_size() const override { return out_.size; }
  inline DType in_type() const override { return get_type<T>(); }
  inline DType out_type() const override { return get_type<T>(); }

  inline virtual const Tensor<T> &out() const { return out_; }
  inline const TensorBase *out_base() const override { return &out_; }

  virtual void forward(const Tensor<T> &input) = 0;

  // noop for layers that doesn't have weights
  inline virtual void load_weights(const Tensor<T> &) {}
  // first one is weights, second one is bias
  inline virtual void load_bias(const Tensor<T> &) {}
  inline virtual bool has_weights() const { return false; }
  inline virtual bool has_bias() const { return false; }
  inline virtual TensorSize weights_size() const { return {0, 0, 0, 0}; }
  inline virtual const Tensor<T> *get_weights() const { return nullptr; }
  inline virtual const Tensor<T> *get_bias() const { return nullptr; }
  inline virtual TensorSize weight_size() const { return {0, 0, 0, 0}; }
  inline virtual TensorSize bias_size() const { return {0, 0, 0, 0}; }

  inline void forward(void *data) override {
    // do not copy the data
    auto t = Tensor<T>(data, in_size(), TensorSize::default_stride(in_size()),
                       false);
    forward(t);
  }

protected:
  TensorSize in_size_;
  Tensor<T> out_;

private:
  template <typename V> inline static DType get_type() {
    static_assert(std::is_fundamental<V>(),
                  "Template type has to be numeric types");
    if (std::is_same<V, int8_t>())
      return DType::Int8;
    else if (std::is_same<V, int16_t>())
      return DType::Int16;
    else if (std::is_same<V, int32_t>())
      return DType::Int32;
    else if (std::is_same<V, int64_t>())
      return DType::Int64;
    else if (std::is_same<V, float>())
      return DType::Float;
    else if (std::is_same<V, double>())
      return DType::Double;
    else
      throw std::runtime_error("Unable to determine types");
  }
};

template <typename T> class Conv2DLayer : public Layer<T> {
public:
  uint32_t filter_size;
  uint32_t num_filters;

  inline Conv2DLayer(const TensorSize &in_size, uint32_t _filter_size,
                     uint32_t _num_filters) :
     Layer<T>(in_size, {in_size.y - _filter_size + 1, in_size.x - _filter_size + 1, _num_filters, in_size.k}),
     filter_size(_filter_size),
     num_filters(_num_filters),
     weights_(_filter_size, _filter_size, in_size.c, _num_filters),
     bias_(1, _num_filters, 1, 1) {}

  inline void set_weight(uint32_t y, uint32_t x, uint32_t c, uint32_t k,
                         T value) {
     weights_(y, x, c, k) = value;
  }

  inline void load_weights(const Tensor<T> &weight) override {
    weights_.load(weight);
  }

  inline TensorSize weights_size() const { return weights_.size; }

  inline void forward(const Tensor<T> &in) override {
     auto & out = Layer<T>::out_;
     // out(y, x, c) += in(y + yy, x + xx, c + cc) * weights_(yy, xx, cc);
     for (uint32_t y = 0; y < out.size.y; ++y) {
        for (uint32_t x = 0; x < out.size.x; ++x) {
           for (uint32_t oc = 0; oc < out.size.c; ++oc) {
              T sum = T();
              for (uint32_t yy = 0; yy < weights_.size.y; ++yy) {
                 for (uint32_t xx = 0; xx < weights_.size.x; ++xx) {
                    for (uint32_t cc = 0; cc < weights_.size.c; ++cc) {
                       sum += in(y + yy, x + xx, cc) * weights_(yy, xx, cc, oc);
                    }
                 }
              }
              out(y, x, oc) = sum + bias_(0, oc, 0);
           }
        }
     }
  }

  inline TensorSize weight_size() const override { return weights_.size; }
  inline TensorSize bias_size() const override { return bias_.size; }
  inline virtual const Tensor<T> *get_weights() const { return &weights_; }
  inline virtual const Tensor<T> *get_bias() const { return &bias_; }

  inline bool has_bias() const override { return true; }
  inline bool has_weights() const override { return true; }
  inline void load_bias(const Tensor<T> &bias) override { bias_.load(bias); }

private:
  Tensor<T> weights_;
  Tensor<T> bias_;
};

template <typename T> class MaxPoolingLayer : public Layer<T> {
public:
  inline explicit MaxPoolingLayer(const TensorSize &in_size,
                                  uint32_t _pool_size) :
     Layer<T>(in_size, {in_size.y - _pool_size + 1, in_size.x - _pool_size + 1, in_size.c, in_size.k}),
     pool_size_(_pool_size) {}

  inline void forward(const Tensor<T> &in) override {
     auto & out = Layer<T>::out_;
     for (uint32_t y = 0; y < out.size.y; ++y) {
        for (uint32_t x = 0; x < out.size.x; ++x) {
           for (uint32_t c = 0; c < out.size.c; ++c) {
              T cur_max = std::numeric_limits<T>::min();
              for (uint32_t yy = 0; yy < pool_size_; ++yy) {
                 for (uint32_t xx = 0; xx < pool_size_; ++xx) {
                    cur_max = std::max(cur_max, in(y + yy, x + xx, c));
                 }
              }
              out(y, x, c) = cur_max;
           }
        }
     }
  }

private:
  uint32_t pool_size_;
};

template <typename T> class FlattenLayer : public Layer<T> {
public:
  inline explicit FlattenLayer(const TensorSize &in_size) : Layer<T>(in_size, {1, in_size.y * in_size.x * in_size.c, 1}) {}

  inline void forward(const Tensor<T> &in) override {
     auto & out = Layer<T>::out_;
     // std::cout << "flatten " << "\n";
     // std::cout << "in_size " << in.size.y << " " << in.size.x << " " << in.size.c << "\n";
     // std::cout << "out_size " << out.size.y << " " << out.size.x << " " << out.size.c << "\n";
     // in.dump(std::cout);
     uint32_t xx = 0, yy = 0, cc = 0;
     for (uint32_t y = 0; y < out.size.y; ++y) {
        for (uint32_t x = 0; x < out.size.x; ++x) {
           for (uint32_t c = 0; c < out.size.c; ++c) {
              if (cc >= in.size.c) {
                 ++xx;
                 cc = 0;
              }
              if (xx >= in.size.x) {
                 ++yy;
                 xx = 0;
              }
              // std::cout << "out_ind " << y << " " << x << " " << c << "\n";
              // std::cout << "in_ind " << yy << " " << xx << " " << cc << "\n\n";
              out(y, x, c) = in(yy, xx, cc);
              // std::cout << "in_tensor " << in(yy, xx, cc) << " " << " end " << "\n";
              // std::cout << "out_tensor " << out(y, x, c) << " " << " end " << "\n";
              // std::cout << y << " " << x << " " << c << "\n";
              // out(y, x, c) = y + x + c;
              // out(y, x, c) = in(y, x, c);
              // out_(0, y * x * c, 0) = in(y, x, c);
              // out_(0, (y * in_size_.x + x) * in_size_.c + c, 0) = in(y, x, c);
              ++cc;
           }
        }
     }
#if 0
     auto n_out = Layer<T>::out();
     std::cout << "out = " << "\n";
     for (uint32_t i = 0; i < 6; i++) {
        std::cout << n_out(0, i, 0) << " ";
     }
#endif
  }
};

template <typename T> class DenseLayer : public Layer<T> {
public:
  inline DenseLayer(const TensorSize &in_size, uint32_t out_size) : 
     Layer<T>(in_size, {in_size.y, out_size, in_size.c, in_size.k}), 
     weights_(in_size.x, out_size, 1, 1),
     bias_(1, out_size, 1, 1) {}

  inline void set_weight(uint32_t y, uint32_t x, uint32_t c, T value) {
     weights_(y, x, c) = value;
  }

  // Expect input to be flattened
  // Expect dim[1] to match up between input and weights
  inline void forward(const Tensor<T> &in) override {
     auto & out = Layer<T>::out_;
     for (uint32_t i = 0; i < in.size.y; ++i) {
        for (uint32_t j = 0; j < weights_.size.x; ++j) {
           T sum = T();
           for (uint32_t k = 0; k < in.size.x; ++k) {
              sum += in(i, k, 0) * weights_(k, j, 0);
           }
           out(i, j, 0) = sum + bias_(0, j, 0);
        }
     }
  }
#if 0
  inline void forward(const Tensor<T> &in) override {
     auto & out = Layer<T>::out_;
     for (uint32_t y = 0; y < in.size.y; ++y) {
        for (uint32_t x = 0; x < in.size.x; ++x) {
           for (uint32_t c = 0; c < in.size.c; ++c) {
              for (uint32_t xx = 0; xx < weights_.size.x; ++xx) {
                 // T sum = T();
                 for (uint32_t yy = 0; yy < weights_.size.y; ++yy) {
                    auto cur_in = in(y, yy, c);
                    auto cur_weight = weights_(yy, xx, 0);
                    out(y, yy, c, xx) += cur_in * cur_weight;
                    // sum += cur_in * cur_weight;
                 }
                 // out(y, x, c, xx) = sum;
                 // out(y, x, c, xx) = sum + bias_(xx, 0, 0);
              }
           }
        }
     }
  }
#endif

  inline bool has_bias() const override { return true; }
  inline bool has_weights() const override { return true; }

  inline void load_weights(const Tensor<T> &weight) override {
    weights_.load(weight);
  }
  inline TensorSize weight_size() const override { return weights_.size; }
  inline TensorSize bias_size() const override { return bias_.size; }

  inline void load_bias(const Tensor<T> &bias) override { bias_.load(bias); }

  inline TensorSize weights_size() const { return weights_.size; }
  inline virtual const Tensor<T> *get_weights() const { return &weights_; }
  inline virtual const Tensor<T> *get_bias() const { return &bias_; }

protected:
  Tensor<T> weights_;
  Tensor<T> bias_;
};

template <typename T> class ActivationLayer : public Layer<T> {
public:
  inline explicit ActivationLayer(const TensorSize &size)
      : Layer<T>(size, size) {}

  inline void forward(const Tensor<T> &in) {}

protected:
  inline virtual T activate_function(T value) { return value; }
};

template <typename T> class ReLuActivationLayer : public ActivationLayer<T> {
public:
  inline explicit ReLuActivationLayer(const TensorSize &size)
      : ActivationLayer<T>(size) {}

  inline void forward(const Tensor<T> &in) {
     auto & out = Layer<T>::out_;
     T zero = T();
     for (uint32_t y = 0; y < out.size.y; ++y) {
        for (uint32_t x = 0; x < out.size.x; ++x) {
           for (uint32_t c = 0; c < out.size.c; ++c) {
              out(y, x, c) = std::max(zero, in(y, x, c));
           }
        }
     }
  }

protected:
  inline T activate_function(T value) override { return 0; }
};

template <typename T> class SigmoidActivationLayer : public ActivationLayer<T> {
public:
  inline explicit SigmoidActivationLayer(const TensorSize &size)
      : ActivationLayer<T>(size) {}

protected:
  inline T activate_function(T value) override { return 0; }
};

#endif // UDNN_LAYER_HH
